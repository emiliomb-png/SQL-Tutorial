import re
import sqlite3
import pandas as pd
from pathlib import Path
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")                      
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------------
# PART I: MATCH PARENTS WITH CHILDREN
# ------------------------------------------------------------------

IND_TXT = "/Users/emilio/Documents/AEDS 6400/Blog Post 6/Data/IND2023ER.txt"   
IND_DO  = "/Users/emilio/Documents/AEDS 6400/Blog Post 6/Data/IND2023ER.do"    
PID_TXT = "/Users/emilio/Documents/AEDS 6400/Blog Post 6/Data/PID23.txt"       
PID_DO  = "/Users/emilio/Documents/AEDS 6400/Blog Post 6/Data/PID23.do"       

EXTRA = {}

# ------------------------------------------------------------------
# 1. LOAD THE FIXED-WIDTH FILES USING THE COLUMN POSITIONS IN THE .do FILE
# ------------------------------------------------------------------
def read_psid(txt_path, do_path, variables):
    """Read selected variables from a PSID ASCII file using its Stata .do layout."""
    do_text = open(do_path, encoding="latin-1").read()

    # Keep only the 'infix ... using' block; entries look like: [long] ER30001 2 - 5
    block = re.split(r"\binfix\b", do_text, maxsplit=1, flags=re.I)[1]
    block = re.split(r"\busing\b", block, maxsplit=1, flags=re.I)[0]
    layout = {name.upper(): (int(a), int(b))
              for name, a, b in re.findall(r"([A-Za-z_]\w*)\s+(\d+)\s*-\s*(\d+)", block)}

    variables = [v.upper() for v in variables]
    missing = [v for v in variables if v not in layout]
    if missing:
        raise KeyError(f"Not found in {do_path}: {missing}")

    # .do positions are 1-indexed inclusive; read_fwf wants 0-indexed half-open
    colspecs = [(layout[v][0] - 1, layout[v][1]) for v in variables]
    return pd.read_fwf(txt_path, colspecs=colspecs, names=variables, header=None)

ind = read_psid(IND_TXT, IND_DO, ["ER30001", "ER30002"] + list(EXTRA.values()))
ind = ind.rename(columns={"ER30001": "id68", "ER30002": "pn",
                          **{v.upper(): k for k, v in EXTRA.items()}})

pid = read_psid(PID_TXT, PID_DO, ["PID2", "PID3", "PID4", "PID5", "PID23", "PID24"])
pid = pid.rename(columns={
    "PID2": "id68", "PID3": "pn",                  # child
    "PID4": "mother_id68", "PID5": "mother_pn",    # birth mother
    "PID23": "father_id68", "PID24": "father_pn",  # birth father
})

# Identifiers must be whole numbers (blank -> 0 = "none")
for df, cols in [(ind, ["id68", "pn"]),
                 (pid, ["id68", "pn", "mother_id68", "mother_pn", "father_id68", "father_pn"])]:
    df[cols] = df[cols].fillna(0).astype("int64")
    
# ------------------------------------------------------------------
# 2. LOAD INTO SQLITE
# ------------------------------------------------------------------
con = sqlite3.connect(":memory:")
ind.to_sql("individuals", con, index=False)
pid.to_sql("parent_ids", con, index=False)

def q(sql):
    return pd.read_sql_query(sql, con)

# ------------------------------------------------------------------
# 3. SELECT: BUILD ONE UNIQUE PERSON KEY (1968 ID * 1000 + person number)
# ------------------------------------------------------------------
extra_select = "".join(f",\n       {k}" for k in EXTRA)

con.executescript(f"""
DROP TABLE IF EXISTS persons;
CREATE TABLE persons AS
SELECT id68 * 1000 + pn AS person_id{extra_select}
FROM individuals;

-- 0 = "no parent": NULLIF turns it into NULL, so a missing ID or person
-- number gives a NULL key and can never match a real person.
DROP TABLE IF EXISTS parent_keys;
CREATE TABLE parent_keys AS
SELECT id68 * 1000 + pn AS person_id,
       NULLIF(mother_id68, 0) * 1000 + NULLIF(mother_pn, 0) AS mother_id,
       NULLIF(father_id68, 0) * 1000 + NULLIF(father_pn, 0) AS father_id
FROM parent_ids;
""")

# ------------------------------------------------------------------
# 4. JOIN (SELF-JOIN): ATTACH MOTHER AND FATHER TO EACH CHILD
# ------------------------------------------------------------------
extra_child = "".join(f",\n       c.{k} AS child_{k}"  for k in EXTRA)
extra_mom   = "".join(f",\n       m.{k} AS mother_{k}" for k in EXTRA)
extra_dad   = "".join(f",\n       f.{k} AS father_{k}" for k in EXTRA)

con.executescript(f"""
DROP TABLE IF EXISTS child_parent;
CREATE TABLE child_parent AS
SELECT pk.person_id AS child_id,
       pk.mother_id AS mother_id,
       pk.father_id AS father_id,
       m.person_id  AS mother_in_ind,
       f.person_id  AS father_in_ind{extra_child}{extra_mom}{extra_dad}
FROM parent_keys AS pk
LEFT JOIN persons AS c ON pk.person_id = c.person_id
LEFT JOIN persons AS m ON pk.mother_id = m.person_id
LEFT JOIN persons AS f ON pk.father_id = f.person_id;
""")

# ------------------------------------------------------------------
# 5. GROUP BY + HAVING: VALIDATE (each child should appear once)
# ------------------------------------------------------------------
dupes = q("""
SELECT child_id, COUNT(*) AS n_rows
FROM child_parent
GROUP BY child_id
HAVING COUNT(*) > 1
""")

# ------------------------------------------------------------------
# 6. AGGREGATION: MATCH RATES
# ------------------------------------------------------------------
print("\nMatch rates:")
print(q("""
SELECT COUNT(*)                                             AS n_children,
       SUM(mother_id IS NOT NULL)                           AS n_mother_id,
       SUM(father_id IS NOT NULL)                           AS n_father_id,
       SUM(mother_id IS NOT NULL AND father_id IS NOT NULL) AS n_both_ids,
       SUM(mother_in_ind IS NOT NULL)                       AS n_mother_in_psid,
       SUM(father_in_ind IS NOT NULL)                       AS n_father_in_psid
FROM child_parent
"""))

# ------------------------------------------------------------------
# 7. GROUP BY + AGGREGATION: CHILDREN PER MOTHER
# ------------------------------------------------------------------
by_mother = q("""
SELECT mother_id, COUNT(*) AS n_children
FROM child_parent
WHERE mother_id IS NOT NULL
GROUP BY mother_id
ORDER BY n_children DESC
""")

# ------------------------------------------------------------------
# 8. FINAL MATCHED SAMPLE: at least one parent observed in the PSID
# ------------------------------------------------------------------
pairs = q("""
SELECT *
FROM child_parent
WHERE mother_in_ind IS NOT NULL OR father_in_ind IS NOT NULL
""")

pairs.to_csv("child_parent_pairs.csv", index=False)

# ------------------------------------------------------------------
#  PART 2: MEASUREMENTS
# ------------------------------------------------------------------

CHILD_PARENT_CSV = "/Users/emilio/Documents/AEDS 6400/Blog Post 6/Results/child_parent_pairs.csv"   # matched sample from part 1
IND_TXT = "/Users/emilio/Documents/AEDS 6400/Blog Post 6/Data/IND2023ER.txt"           
IND_DO  = "/Users/emilio/Documents/AEDS 6400/Blog Post 6/Data/IND2023ER.do"
FAM_DIR = "/Users/emilio/Documents/AEDS 6400/Blog Post 6/Data/psid_family_files"                

# ------------------------------------------------------------------
# MEASUREMENT CHOICES
# ------------------------------------------------------------------
PARENT_AGES, PARENT_CENTER, PARENT_MIN_YEARS = (15, 19), 17, 2   # child's age when parental income is measured
CHILD_AGES,  CHILD_CENTER,  CHILD_MIN_YEARS  = (30, 40), 35, 3   # child's age when child income is measured
FLOOR = 1000        # drop incomes below this so logs are defined
BASE_YEAR = 2016    # express dollars in this year's prices
CPI = {
    1983: 99.6,   1984: 103.9,  1985: 107.6,  1986: 109.6,  1987: 113.6,
    1988: 118.3,  1989: 124.0,  1990: 130.7,  1991: 136.2,  1992: 140.3,
    1993: 144.5,  1994: 148.2,  1995: 152.4,  1996: 156.9,  1997: 160.5,
    1998: 163.0,  1999: 166.6,  2000: 172.2,  2001: 177.1,  2002: 179.9,
    2003: 184.0,  2004: 188.9,  2005: 195.3,  2006: 201.6,  2007: 207.342,
    2008: 215.303, 2009: 214.537, 2010: 218.056, 2011: 224.939, 2012: 229.594,
    2013: 232.957, 2014: 236.736, 2015: 237.017, 2016: 240.007,
}            # {income year: CPI-U annual average}, e.g. {1983: ..., ..., 2016: ...}. Empty = nominal dollars

# interview year: (family income var [family file], family interview no. [individual file], year born [individual file])
# The income variable refers to the previous calendar year.
WAVES = {
    1984: ("V11022", "ER30429", "ER30434"), 1985: ("V12371", "ER30463", "ER30468"),
    1986: ("V13623", "ER30498", "ER30503"), 1987: ("V14670", "ER30535", "ER30540"),
    1988: ("V16144", "ER30570", "ER30575"), 1989: ("V17533", "ER30606", "ER30611"),
    1990: ("V18875", "ER30642", "ER30647"), 1991: ("V20175", "ER30689", "ER30694"),
    1992: ("V21481", "ER30733", "ER30738"), 1993: ("V23322", "ER30806", "ER30811"),
    1994: ("ER4153",  "ER33101", "ER33106"), 1995: ("ER6993",  "ER33201", "ER33206"),
    1996: ("ER9244",  "ER33301", "ER33306"), 1997: ("ER12079", "ER33401", "ER33406"),
    1999: ("ER16462", "ER33501", "ER33506"), 2001: ("ER20456", "ER33601", "ER33606"),
    2003: ("ER24099", "ER33701", "ER33706"), 2005: ("ER28037", "ER33801", "ER33806"),
    2007: ("ER41027", "ER33901", "ER33906"), 2009: ("ER46935", "ER34001", "ER34006"),
    2011: ("ER52343", "ER34101", "ER34106"), 2013: ("ER58152", "ER34201", "ER34206"),
    2015: ("ER65349", "ER34301", "ER34307"), 2017: ("ER71426", "ER34501", "ER34506"),
}

# ------------------------------------------------------------------
# 1. LOAD THE PSID FILES (fixed-width text; column positions come from the .do file)
# ------------------------------------------------------------------
def layout(do_path):
    text = open(do_path, encoding="latin-1").read()
    block = re.split(r"\binfix\b", text, maxsplit=1, flags=re.I)[1]
    block = re.split(r"\busing\b", block, maxsplit=1, flags=re.I)[0]
    return {n.upper(): (int(a), int(b))
            for n, a, b in re.findall(r"([A-Za-z_]\w*)\s+(\d+)\s*-\s*(\d+)", block)}

def read_psid(txt, do, variables):
    lay = layout(do)
    specs = [(lay[v][0] - 1, lay[v][1]) for v in variables]
    return pd.read_fwf(txt, colspecs=specs, names=variables, header=None)

# find each wave's family file (.txt and .do) anywhere inside FAM_DIR
found = {}
for p in Path(FAM_DIR).expanduser().rglob("*"):
    m = re.match(r"fam(\d{4})", p.name.lower())
    if m and p.suffix.lower() in (".txt", ".do"):
        found.setdefault(int(m.group(1)), {})[p.suffix.lower()] = str(p)
YEARS = [y for y in WAVES if all(e in found.get(y, {}) for e in (".txt", ".do"))]
print("Waves used:", YEARS)

# Individual file: for each wave, the person's family interview number and birth year
cols, names = ["ER30001", "ER30002"], ["id68", "pn"]
for y in YEARS:
    cols += WAVES[y][1:]
    names += [f"fam_{y}", f"byear_{y}"]
ind = read_psid(IND_TXT, IND_DO, cols)
ind.columns = names
ind = ind.fillna(0).astype("int64")

# Family files: family interview number (2nd variable in each file) and total family income
frames = []
for y in YEARS:
    txt, do = found[y][".txt"], found[y][".do"]
    id_var = list(layout(do))[1]
    df = read_psid(txt, do, [id_var, WAVES[y][0]])
    df.columns = ["fam_id", "income"]
    frames.append(df.dropna().astype({"fam_id": "int64"}).assign(interview_year=y))
family_raw = pd.concat(frames)

if not CPI:
    print("WARNING: CPI is empty, so dollars are nominal (not comparable across years).")
    CPI = {y: 1.0 for y in range(1900, 2031)}
assert all(y - 1 in CPI for y in YEARS) and BASE_YEAR in CPI, "CPI is missing some years"

# ------------------------------------------------------------------
# 2. PUT EVERYTHING IN SQLITE
# ------------------------------------------------------------------
con = sqlite3.connect(":memory:")
ind.to_sql("ind_raw", con, index=False)
family_raw.to_sql("family_raw", con, index=False)
pd.DataFrame({"year": list(CPI), "cpi": list(CPI.values())}).to_sql("cpi", con, index=False)
pd.read_csv(CHILD_PARENT_CSV)[["child_id", "mother_id", "father_id"]].to_sql("child_parent", con, index=False)

def q(sql):
    return pd.read_sql_query(sql, con)

# ------------------------------------------------------------------
# 3. BUILD THE INCOME TABLES IN SQL
# ------------------------------------------------------------------
# UNION ALL turns the wide file (one column per wave) into one row per person per wave
union = "\nUNION ALL\n".join(
    f"SELECT id68 * 1000 + pn AS person_id, {y} AS interview_year, fam_{y} AS fam_id, byear_{y} AS byear "
    f"FROM ind_raw" for y in YEARS)

con.executescript(f"""
CREATE TABLE person_year AS
SELECT * FROM ({union}) WHERE fam_id > 0;          -- fam_id 0 = not in the survey that wave

-- one birth year per person: average of the valid reports across waves
CREATE TABLE birth AS
SELECT person_id, CAST(ROUND(AVG(byear)) AS INTEGER) AS birth_year
FROM person_year WHERE byear BETWEEN 1850 AND 2025
GROUP BY person_id;

-- family income by wave in base-year dollars (income year = interview year - 1)
CREATE TABLE family_income AS
SELECT f.interview_year, f.fam_id, f.interview_year - 1 AS income_year,
       f.income * {CPI[BASE_YEAR]} / c.cpi AS income_real
FROM family_raw AS f JOIN cpi AS c ON c.year = f.interview_year - 1;

CREATE INDEX i1 ON family_income(interview_year, fam_id);   -- indexes speed up the big joins
CREATE INDEX i2 ON person_year(person_id);

-- (child, parent) pairs from the step 1 match: mothers and fathers stacked
CREATE TABLE parent_links AS
SELECT child_id, mother_id AS parent_id FROM child_parent WHERE mother_id IS NOT NULL
UNION
SELECT child_id, father_id              FROM child_parent WHERE father_id IS NOT NULL;

-- parental income: income of the parents' own family unit(s) in each year,
-- labelled with the CHILD's age (averaged if the parents were in different units)
CREATE TABLE parent_year AS
SELECT pl.child_id, f.income_year - b.birth_year AS age, AVG(f.income_real) AS income
FROM parent_links AS pl
JOIN person_year   AS py ON py.person_id = pl.parent_id
JOIN family_income AS f  ON f.interview_year = py.interview_year AND f.fam_id = py.fam_id
JOIN birth         AS b  ON b.person_id = pl.child_id
GROUP BY pl.child_id, f.income_year, b.birth_year;

-- child income: income of the child's own family unit in each year, with the child's age
CREATE TABLE child_year AS
SELECT py.person_id AS child_id, f.income_year - b.birth_year AS age, f.income_real AS income
FROM person_year AS py
JOIN family_income AS f ON f.interview_year = py.interview_year AND f.fam_id = py.fam_id
JOIN birth         AS b ON b.person_id = py.person_id
WHERE py.person_id IN (SELECT child_id FROM child_parent);
""")

# Check: nearly every person-wave should find its family record
print("Person-waves with no family record (should be ~0):",
      q("""SELECT COUNT(*) AS n FROM person_year p
           LEFT JOIN family_income f ON f.interview_year = p.interview_year AND f.fam_id = p.fam_id
           WHERE f.fam_id IS NULL""").n[0])

# ------------------------------------------------------------------
# 4. SINGLE-YEAR vs MULTI-YEAR MEASURES (same SQL for parents and children)
# ------------------------------------------------------------------
def measures(table, lo, hi, center, min_years):
    # rank each person's observed years by closeness to the center age:
    #   rank 1 -> SINGLE-YEAR measure;  average of all years in the window -> MULTI-YEAR measure
    return q(f"""
    WITH win AS (SELECT child_id, age, income FROM {table} WHERE age BETWEEN {lo} AND {hi}),
    ranked AS (
        SELECT child_id, income,
               ROW_NUMBER() OVER (PARTITION BY child_id ORDER BY ABS(age - {center}), age) AS rk,
               COUNT(*)     OVER (PARTITION BY child_id) AS n_years
        FROM win)
    SELECT child_id,
           AVG(CASE WHEN rk = 1 THEN income END) AS single,
           AVG(income)                           AS multi
    FROM ranked
    GROUP BY child_id
    HAVING MAX(n_years) >= {min_years}""")

parents  = measures("parent_year", *PARENT_AGES, PARENT_CENTER, PARENT_MIN_YEARS)
children = measures("child_year",  *CHILD_AGES,  CHILD_CENTER,  CHILD_MIN_YEARS)
df = parents.merge(children, on="child_id", suffixes=("_parent", "_child"))
df = df[(df[["single_parent", "multi_parent", "single_child", "multi_child"]] >= FLOOR).all(axis=1)]
print(f"\nChildren with all four income measures: {len(df):,}")   # same children in every estimate

# ------------------------------------------------------------------
# 5. IGE: slope of ln(child income) on ln(parent income)
# ------------------------------------------------------------------
def ige(d, y, x):
    fit = sm.OLS(np.log(d[y]), sm.add_constant(np.log(d[x]))).fit()
    return round(fit.params.iloc[1], 3), round(fit.bse.iloc[1], 3)

rows = [(p, c, *ige(df, c, p), len(df))
        for p in ["single_parent", "multi_parent"] for c in ["single_child", "multi_child"]]
print("\nIGE: single-year vs multi-year income")
print(pd.DataFrame(rows, columns=["parent income", "child income", "IGE", "SE", "N"]).to_string(index=False))

# ------------------------------------------------------------------
# 6. LIFE-CYCLE BIAS: IGE when child income is measured at ONE age
# ------------------------------------------------------------------
cy = q("SELECT child_id, age, income FROM child_year").merge(parents[["child_id", "multi"]], on="child_id")
cy = cy[(cy.income >= FLOOR) & (cy.multi >= FLOOR)]
rows = [(a, len(d), round(d.income.mean()), *ige(d, "income", "multi"))
        for a in range(25, 46, 5) for d in [cy[cy.age == a]] if len(d) >= 30]
print("\nIGE by the child's age when income is measured (parent income = multi-year average)")
print(pd.DataFrame(rows, columns=["child age", "N", "mean child income", "IGE", "SE"]).to_string(index=False))

# ---- Markdown tables for the post ----
def md(frame):
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "|" + "---|" * len(frame.columns)]
    for _, r in frame.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(lines)

# Table 1: IGE, single-year vs multi-year
t1 = pd.DataFrame({
    "Parent income": ["Single year", "Multi-year"],
    "Child: single year": [ige(df, "single_child", p)[0] for p in ["single_parent", "multi_parent"]],
    "Child: multi-year":  [ige(df, "multi_child",  p)[0] for p in ["single_parent", "multi_parent"]]})
print(md(t1))

# Table 2: IGE and mean income by the child's age
ages = [a for a in range(25, 46, 5) if (cy.age == a).sum() >= 30]
t2 = pd.DataFrame({
    "Child age": ages,
    "IGE": [ige(cy[cy.age == a], "income", "multi")[0] for a in ages],
    "Mean income": [f"${cy[cy.age == a].income.mean():,.0f}" for a in ages]})
print(md(t2.set_index("Child age").T.reset_index().rename(columns={"index": "Child age"})))
