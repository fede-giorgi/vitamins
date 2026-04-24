import re

SYSTEM_MSG = """
You are a strict binary classifier for emergency department (ED) narratives.
You will receive a batch of narratives formatted as a JSON dictionary: {"id_1": "text_1", "id_2": "text_2", ...}

OUTPUT FORMAT: You MUST return a JSON array containing EXACTLY ONE dictionary for each narrative provided.
Example:
[
  {"id": "id_1", "reason": "short reason", "label": 0},
  {"id": "id_2", "reason": "short reason", "label": 1}
]

TASK:
Classify whether the narrative involves exposure/ingestion/overdose/adverse reaction to a STRICTLY HARMLESS VITAMIN.

LABEL 1 (POSITIVE) - INCLUSION RULES:
- label=1 ONLY if the narrative explicitly involves a clear, traditional VITAMIN or fish oil.
  Examples: "multivitamin", "childrens gummy multivitamins", "vitamin D", "vitamin C", "fish oil".
- SAFE CO-INGESTIONS: If the patient ingested a vitamin alongside another SAFE substance (e.g., "ate vitamin C and a multivitamin", "drank juice and took vitamins"), label it 1.
- MISSPELLINGS & TYPOS: ED narratives are badly written. You MUST tolerate and accept typos for vitamins (e.g., "mutivitiamins", "vitmin", "gummis").
- REDACTED BRANDS (***): If the word "vitamin" is present but the brand is redacted (e.g., "***VITAMINS", "VITAMINS ***"), label it 1. The asterisks just hide the brand name, it is still a vitamin.

LABEL 0 (NEGATIVE) - EXCLUSION RULES (Highest Priority):
- EXCLUSIONS OVERRIDE INCLUSIONS: If ANY of the following are true, you MUST label=0, even if the word "vitamin" is present.
  1. IRON & PRENATAL EXCEPTION: Any explicit mention of iron (e.g., "WITH IRON", "IRON", "Fe", "ferrous"). AND ALL "PRENATAL" or "PRENAT" vitamins MUST be excluded (label=0) because they implicitly contain toxic amounts of iron. *Note: Standard "multivitamins" are safe unless iron is stated.*
  2. DANGEROUS CO-INGESTIONS: If the patient ingested a safe vitamin ALONG WITH a dangerous/prescription medication (e.g., beta blocker, Tylenol, potassium pill, cough med), label it 0. The ED visit is driven by the dangerous drug, not the vitamin.
  3. REDACTED MULTI-DRUG INGESTIONS: If the narrative describes taking mixed medications/pill boxes but the names of the other drugs are redacted alongside a vitamin (e.g., "UNCLE'S MEDS... ***, ***, VITAMINS"), label it 0. You MUST assume the redacted substances are dangerous co-ingestions.
  4. VITAMINS GIVEN AS TREATMENT: If the vitamin was ADMINISTERED by doctors/hospital as a treatment for something else (e.g., "GIVEN THIAMINE", "GIVEN MVI", "Rx vitamins" for alcohol/whiskey intoxication), label it 0. We ONLY want accidental poisonings/ingestions of vitamins, not medical treatments.
  5. NON-VITAMIN SUPPLEMENTS: Melatonin, diet pills, fat loss pills, herbal supplements, botanicals, creatine, protein.
  6. CANNABIS: CBD, THC, marijuana gummies, weed.
  7. FULLY REDACTED: ONLY if the *entire* substance is unknown AND the word 'vitamin' is missing (e.g., "ate chewable ***"), label it 0.
  8. PRESCRIPTION/OTC DRUGS: Any standard medication.
  9. DRUG TYPOS: Be highly vigilant for misspelled drugs (e.g., "xanTax", "tylenol", "ibuprofin").
  10. HOUSEHOLD/COSMETICS TOXINS: Shampoos, lotions, creams, or soaps that happen to have "vitamin" in their name (e.g., "shampoo with vitamin E").

GENERAL RULES:
- ACCEPT GENERIC TERMS (BENEFIT OF THE DOUBT): If the text mentions generic "vitamins", "gummy vitamins", "vitamin pills", or "childrens vitamins" without specifying the exact type, ASSUME THEY ARE SAFE (label=1) as long as NO exclusion keywords (iron, melatonin, drugs, etc.) are present. Do not punish the narrative for being vague.
- Reason must cite the key phrase that triggered your decision.
""".strip()

# Few-shot examples WITH reasons (include hard negatives for new clinical rules)
FEW_SHOTS = [
    # Positives
    ("4YOM WITH ABD PAIN S/P EATING HANDFUL OF CHILDRENS GUMMY MULTIVITAMINS", 1, "Explicit mention of 'childrens gummy multivitamins'. This is a standard safe vitamin."),
    ("PT INGESTED 10 MUTIVITIAMINS THEY WERE IN A PLASTIC BAG", 1, "Mentions 'mutivitiamins' (misspelled). Assumed safe unless iron is explicitly mentioned."),
    ("16MOF GOT INTO VITAMINS 4 DAYS AGO, INGESTED UNKNOWN NUMBER OF ***,***,***", 1, "The generic word 'VITAMINS' is present. The asterisks merely hide the brand. Assumed safe."),
    ("PT ATE ***VITAMINS", 1, "The word 'VITAMINS' is attached to the redaction. It is a vitamin, asterisks just hide the brand."),
    ("3YOF INGESTION OF 20 MULTIVITAMIN GUMMIES, *** GUMMIES.", 1, "Clear mention of 'multivitamin gummies'. The presence of redacted '***' does not negate the vitamin ingestion."),

    # Hard negatives (Drugs, Cosmetics, Melatonin, Cannabis)
    ("23MOM SWALLOWED *** SHAMPOO LUXURIOUS MOISTURE & VITAMINE", 0, "Mentions 'shampoo'. Cosmetics and household chemicals are excluded, even if they contain the word vitamin."),
    ("ADULT TOOK MELATONIN GUMMIES; DIZZY.", 0, "Mentions 'melatonin', which is an excluded non-vitamin supplement."),
    ("3 YOF FOUND EATING CBD GUMMY.", 0, "Mentions 'CBD', cannabis products are excluded."),
    ("3YF FD WITH OPEN BOTTLE OF CHILDREN'S CHEWABLE ***.", 0, "The specific substance is totally redacted ('***'). We cannot confirm it is a vitamin."),

    # Co-ingestion overrides (Must be 0)
    ("INGESTED BETA BLOCKER PILL, MULTI VITAMIN AND POTASSIUM PILL FROM GRANDMAS DAILY PILL MINDER", 0, "Co-ingestion. A 'MULTI VITAMIN' is mentioned, but was ingested alongside dangerous medications ('BETA BLOCKER', 'POTASSIUM'). Excluded."),
    ("PATIENT INGESTED 2-3 DAYS WORTH OF UNCLE'S MEDS FROM PILL BOX, INCLUDED***, ***, ***, ***, VITAMINS; CHARCOAL ACTIVATED", 0, "Multi-drug ingestion from a pill box where other medications are redacted ('***'). Must assume dangerous co-ingestion. Excluded."),

    # Vitamin as Treatment overrides (Must be 0)
    ("PT FOUND UNRESPONSIVE ON THE FLOOR AT HOME LYING NEXT TO AN EMPTY BOTTLE OF WHISKEY BAC- 299, GIVEN FLUIDS, THIAMINE MVI", 0, "Mentions 'THIAMINE' and 'MVI' (multivitamin), but they were GIVEN as treatment by the hospital for alcohol ingestion. Not an accidental vitamin ingestion."),

    # Iron exceptions (Must be 0)
    ("PATIENT INGESTED 1/2 BOTTLE *** VITAMINS 10 MG IRON", 0, "Explicitly mentions 'IRON' in the formulation."),
    ("PRENATAL VITAMINS WITH IRON INGESTION.", 0, "Explicitly mentions 'PRENATAL' and 'IRON'. Both trigger exclusion."),
    ("PT INGESTED 1 *** PRENAT VITAMIN", 0, "Mentions 'PRENAT VITAMIN'. Prenatal vitamins implicitly contain high iron and must be excluded.")
]

EXCLUSION_RULES = [
    (re.compile(r"\b(cbd|thc|marijuana|weed|cannabis|hemp)\b", re.IGNORECASE), 
     "Mentions cannabis product."),
    (re.compile(r"\b(melatonin)\b", re.IGNORECASE), 
     "Mentions melatonin (non-vitamin supplement)."),
    (re.compile(r"\b(diet pill|fat loss|weight loss)\b", re.IGNORECASE), 
     "Mentions weight-loss/diet supplement.")
]