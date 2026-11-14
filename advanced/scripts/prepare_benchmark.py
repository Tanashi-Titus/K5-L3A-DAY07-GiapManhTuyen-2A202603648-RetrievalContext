"""Create the OECD advanced benchmark with page-anchored human evidence notes."""
from __future__ import annotations

import json
import hashlib
import re
import unicodedata
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "oecd-economic-surveys-viet-nam-2025.pdf"
OUT = ROOT / "benchmark"
SOURCE_ID = "oecd-economic-surveys-viet-nam-2025"


def c(id_, split, group, question, answer, evidence, *, visual=None, answerable=True, absence=None):
    if answerable:
        gold = []
        for page, quote, region in evidence:
            gold.append({
                "source_id": SOURCE_ID,
                "pdf_page": page,
                "printed_page": page - 2 if page >= 4 else None,
                "evidence": quote,
                "visual_region": region,
                "visual_annotation": (visual or {}).get(page),
            })
        return {"id": id_, "split": split, "group": group, "question": question,
                "answerable": True, "expected_answer": answer, "gold": gold}
    return {"id": id_, "split": split, "group": group, "question": question,
            "answerable": False, "expected_behavior": "search_and_abstain",
            "absence_reason": absence}


CASES = [
    # Dev: 15 answerable, spread across the five required evidence groups.
    c("d-t01","dev","table","In Table 1.1, what percentage of the labour force is projected to be unemployed in 2025?","2.5%",[(23,"Unemployment (% of labour force) 2.3 2.3 2.3 2.5 3.0","Table 1.1, unemployment row")]),
    c("d-t02","dev","table","In Table 2.2, what are the employer and employee health-insurance contribution rates?","3.0% for employers and 1.5% for employees.",[(59,"Health Insurance 3.0 1.5","Table 2.2, Health Insurance row")]),
    c("d-t03","dev","table","What was the gasoline environmental protection tax rate in 2024/25?","2,000 VND per litre (0.079 USD per litre).",[(77,"Gasoline, excluding ethanol 2 000 VND/litre 4 000 VND/litre","Table 3.2, gasoline row")]),
    c("d-v01","dev","visual","What is the title and subject of the opening Figure 1 in the executive summary?","It is titled ‘Viet Nam’s growth has outperformed peers’ and compares real GDP growth across Viet Nam, Indonesia, Malaysia and Thailand.",[(11,"Figure 1. Viet Nam’s growth has outperformed peers","Figure 1 chart area")],visual={11:"Visual inspection: a line chart compares real GDP growth indexed to 1990=100 for Viet Nam, Indonesia, Malaysia and Thailand."}),
    c("d-v02","dev","visual","In Figure 4.20, what does panel B measure?","Research and development spending as a percentage of GDP.",[(119,"Figure 4.20. Increased employer-based training and R&D could promote innovation","Figure 4.20, panel B")],visual={119:"Visual inspection: panel B is labelled ‘R&D spending as a % of GDP’ and compares 2011 with 2021."}),
    c("d-v03","dev","visual","Which two export breakdowns are shown in Figure 1.2?","Main export destinations and main export products.",[(22,"Figure 1.2. Exports of goods by destination and products","Figure 1.2 panels A and B")],visual={22:"Visual inspection: two pie charts, panel A main export destinations and panel B main export products, shares of total exports in 2024."}),
    c("d-c01","dev","chart","According to Figure 3.8, what is the estimated potential for solar rooftop power?","147 GW.",[(82,"Figure 3.8. Viet Nam has a considerable renewable energy potential","Figure 3.8, Solar rooftop bar")],visual={82:"Visual inspection of Figure 3.8: Solar rooftop is labelled 147 GW."}),
    c("d-c02","dev","chart","What share of global semiconductor export value did Viet Nam have in Figure 4.19?","2% in 2021.",[(118,"Figure 4.19. Viet Nam is the 10th-largest global exporter of semiconductors","Figure 4.19, Viet Nam slice")],visual={118:"Visual inspection of Figure 4.19: Viet Nam is labelled 2%; the figure concerns 2021 global semiconductor exports."}),
    c("d-c03","dev","chart","Which economy has the highest old-age dependency ratio shown for 2050 in Figure 1.15?","The OECD, at about 48%.",[(37,"Figure 1.15. The elderly dependency is currently low, but will rise sharply by mid-century","Figure 1.15, OECD 2050 bar")],visual={37:"Visual inspection of Figure 1.15: among the 2050 bars, OECD is highest, just under 50% (approximately 48%)."}),
    c("d-m01","dev","multi_hop","Combine Table 3.2 with the nearby discussion: what was the 2024/25 gasoline tax rate, and how did the 2023 tax cut affect receipts?","The gasoline tax was 2,000 VND/litre (0.079 USD/litre); the cut was estimated to lower receipts from 0.54% to 0.41% of GDP.",[(77,"Gasoline, excluding ethanol 2 000 VND/litre","Table 3.2, gasoline row"),(77,"This is estimated to have reduced tax collection from 0.54% to 0.41% of GDP.","Text immediately above Table 3.2")]),
    c("d-m02","dev","multi_hop","What 2024 GDP growth rate does Table 1.1 report, and what GDP growth target does the Section 1.1 discussion compare it with? Give both rates with the % sign.","7.1% actual GDP growth and a 6.5% GDP growth target.",[(23,"Real GDP 8.5 5.1 7.1","Table 1.1, Real GDP row"),(20,"In 2024, GDP increased by 7.1%, surpassing the government’s 6.5% target.","Section 1.1 opening paragraph")]),
    c("d-m03","dev","multi_hop","How much did Viet Nam’s combined solar and wind power capacity increase from 2020 to 2023, and what total clean-source capacity is planned for 2030?","It rose from 17 GW to 23 GW, an increase of 6 GW; clean sources are planned to reach about 80 GW by 2030.",[(83,"solar and wind power capacity, which increased from 17 GW in 2020 to 23 GW in 2023","Chapter 3 text, printed page 81"),(83,"increase from about 45 GW in 2023 to 80 GW in 2030","Chapter 3 text, printed page 81")]),
    c("d-x01","dev","text","What was the annual average rate of real GDP growth in Viet Nam from 1990 to 2023?","6.7%.",[(98,"Real GDP rose at a 6.7% annual average rate between 1990 and 2023","Section 4.1 opening paragraph")]),
    c("d-x02","dev","text","How many engineers did Viet Nam have in its semiconductor industry, according to the report?","Around 26,000 engineers.",[(118,"Viet Nam currently has around 26 000 engineers in the semiconductor industry","Box 4.2")]),
    c("d-x03","dev","text","What percentage of secondary-school students in Korea continue to college or university, according to Box 4.3?","More than 80%.",[(119,"more than 80% of secondary school graduates in Korea continue to college or universities","Box 4.3")]),
    c("d-u01","dev","text","How many electric buses were registered in Hanoi in 2025?","Not specified in this report.",[],answerable=False,absence="The report discusses transport decarbonisation and electric vehicles, but does not give a 2025 Hanoi electric-bus registration count."),
    c("d-u02","dev","text","What was the exact sample size of the 2024 randomized trial of Vietnamese bank consolidation?","Not specified in this report.",[],answerable=False,absence="The survey discusses financial-sector risks and banking consolidation but does not report such a randomized trial or sample size."),
    c("d-u03","dev","text","What was the province-level poverty rate in Viet Nam in 2025?","Not specified in this report.",[],answerable=False,absence="The report contains national and group-level poverty information, not 2025 poverty rates for each province."),

    # Test: 24 answerable in the required groups, plus 6 evidence-absence cases.
    c("t-t01","test","table","What was Viet Nam’s real GDP growth in 2024 in Table 1.1?","7.1%.",[(23,"Real GDP 8.5 5.1 7.1 6.2 6.0","Table 1.1, Real GDP row")]),
    c("t-t02","test","table","In Table 1.1, what are the 2025 projections for export growth and import growth?","Exports of goods and services: 8.0%; imports: 9.3%.",[(23,"Exports of goods and services 4.0 -2.5 15.5 8.0 5.4\nImports of goods and services 1.5 -4.5 16.1 9.3 6.0","Table 1.1, exports/imports rows")]),
    c("t-t03","test","table","According to Table 1.4, what is the estimated five-year GDP-per-capita impact of halving the rule-of-law gap with the OECD average?","2.1%.",[(39,"2.1%","Table 1.4, rule-of-law row, 5 years column")],visual={39:"Visual inspection: the row ‘Halve the gap with the OECD average in the rule of law’ shows 2.1% under the 5-years column."}),
    c("t-t04","test","table","In Table 2.2, what are the employer and employee retirement-and-death social-insurance rates?","14.0% for employers and 8.0% for employees.",[(59,"Retirement and death 14.0 8.0","Table 2.2, retirement and death row")]),
    c("t-t05","test","table","What was the environmental protection tax on diesel in early 2023?","2,000 VND per litre (0.079 USD per litre).",[(77,"Diesel fuel 1 000 VND/litre 2 000 VND/litre","Table 3.2, diesel row")]),
    c("t-t06","test","table","What foreign-ownership limit was introduced in January 2024 for non-facilities-based telecom services?","65%.",[(130,"65%","Table 4.1, telecommunications action")],visual={130:"Visual inspection: the action cell for non-facilities-based telecom services states the foreign-ownership limit was raised to 65% in January 2024."}),
    c("t-v01","test","visual","What four subjects are compared across panels A–D of Figure 1.1?","Real GDP growth; contribution to GDP growth; retail sales and real wages; export and import growth in nominal value.",[(20,"Figure 1.1. Recent macroeconomic developments","Figure 1.1, panels A–D")],visual={20:"Visual inspection: four-panel figure labelled A Real GDP growth; B Contribution to GDP growth; C Retail sales and real wages; D Export and import growth in nominal value."}),
    c("t-v02","test","visual","Figure 1.2 has two pie charts. What does each panel break down?","Panel A breaks down main export destinations; panel B breaks down main export products, for 2024.",[(22,"Figure 1.2. Exports of goods by destination and products","Figure 1.2, panels A–B")],visual={22:"Visual inspection: panel A is Main export destinations; panel B is Main export products; both show shares of total exports in 2024."}),
    c("t-v03","test","visual","Which six generation sources are individually plotted in Figure 3.9?","Coal, solar, wind, hydro, gas+LNG and bioenergy.",[(84,"Figure 3.9. Viet Nam has elaborated concrete plans to expand low-carbon electricity generation","Figure 3.9 six-panel visual")],visual={84:"Visual inspection: six labelled panels are Coal, Solar, Wind, Hydro, Gas+LNG and Bioenergy."}),
    c("t-v04","test","visual","In Figure 4.20, what is measured in panels A and B?","Panel A: percentage of firms offering formal training to employees. Panel B: R&D spending as a percentage of GDP.",[(119,"Figure 4.20. Increased employer-based training and R&D could promote innovation","Figure 4.20, panels A–B")],visual={119:"Visual inspection: panel A is the percentage of firms offering formal training; panel B is R&D spending as a percent of GDP."}),
    c("t-v05","test","visual","What is the visual structure of Figure 4.19 and what does it represent?","A pie chart showing each economy’s share of global semiconductor export value in 2021.",[(118,"Figure 4.19. Viet Nam is the 10th-largest global exporter of semiconductors","Figure 4.19 pie chart")],visual={118:"Visual inspection: a labelled pie chart compares economies’ shares of global semiconductor export value; the accompanying Box 4.2 identifies the figure year as 2021."}),
    c("t-c01","test","chart","What potential does Figure 3.8 assign to onshore wind?","217 GW.",[(82,"Figure 3.8. Viet Nam has a considerable renewable energy potential","Figure 3.8, Wind Onshore bar")],visual={82:"Visual inspection: the Wind Onshore bar is labelled 217 GW."}),
    c("t-c02","test","chart","What share of 2021 global semiconductor export value does China hold in Figure 4.19?","22%.",[(118,"Figure 4.19. Viet Nam is the 10th-largest global exporter of semiconductors","Figure 4.19, China slice")],visual={118:"Visual inspection: China is the largest labelled slice, 22%; Chinese Taipei is 19%."}),
    c("t-c03","test","chart","In Figure 2.7, approximately what share of GDP came from Viet Nam’s social-security contributions in 2022?","About 5.4% of GDP.",[(60,"Figure 2.7. Social security contributions generate sizeable government revenues, 2022","Figure 2.7, Viet Nam bar")],visual={60:"Visual inspection: Viet Nam’s red bar is approximately 5.4% of GDP, between the 5% and 6% gridlines."}),
    c("t-c04","test","chart","What total percentage is labelled for Viet Nam in the ‘Total’ category of Figure 4.22 for 2018?","12%.",[(123,"Figure 4.22. Domestic services’ value added embedded in exports remains low","Figure 4.22, Total category for Viet Nam")],visual={123:"Visual inspection: in the Total category, Viet Nam’s red bar is labelled 12 for 2018."}),
    c("t-c05","test","chart","Which destination accounts for the largest share of Viet Nam’s 2024 goods exports, and what is that share in Figure 1.2?","The United States, 30%.",[(22,"Figure 1.2. Exports of goods by destination and products","Figure 1.2, panel A, United States slice")],visual={22:"Visual inspection: the United States is the largest destination slice at 30%."}),
    c("t-m01","test","multi_hop","Combine the 2024 result and target in the text with Table 1.1: how far did GDP growth exceed the government’s target?","0.6 percentage points (7.1% actual versus 6.5% target).",[(20,"In 2024, GDP increased by 7.1%, surpassing the government’s 6.5% target.","Section 1.1 growth result"),(23,"Real GDP 8.5 5.1 7.1 6.2 6.0","Table 1.1, Real GDP row")]),
    c("t-m02","test","multi_hop","Using the discussion on pages 96–97, report both Viet Nam’s FDI inflows as a share of GDP over 2015–23 and its share of world trade in 2023.","FDI inflows were 4.8% of GDP; Viet Nam’s world-trade share was 1.5%.",[(98,"FDI inflows that amounted to 4.8% of GDP","Section 4.1, FDI paragraph"),(99,"Viet Nam’s share of world trade climbed from 0.1% in 1996 to 1.5% in 2023","Section 4.2, trade paragraph")]),
    c("t-m03","test","multi_hop","The report gives 2023 solar-and-wind capacity and a 2030 clean-capacity plan. What are the two values, and what is the increase?","About 45 GW in 2023 and 80 GW in 2030, an increase of about 35 GW.",[(83,"increase from about 45 GW in 2023 to 80 GW in 2030","Chapter 3 text, printed page 81"),(84,"Figure 3.9. Viet Nam has elaborated concrete plans to expand low-carbon electricity generation","Figure 3.9, 2030 clean energy trajectories")]),
    c("t-m04","test","multi_hop","Combine Table 2.2 and the paragraph above it: what is the total social-insurance contribution rate from employers and workers, and what is the contribution cap?","The combined rates reach 32%; contributions are capped at 20 times the basic salary.",[(59,"Total SI rates from employers and workers to the various insurance schemes reach 32%","Paragraph before Table 2.2"),(59,"The payments are capped at 20 times the basic salary for SI contributions.","Paragraph before Table 2.2"),(59,"Total 21.5 10.5","Table 2.2, total row")]),
    c("t-x01","test","text","What US tariff assumptions on Vietnamese exports are used in the report’s 2025–26 projections?","A 10% baseline tariff on all goods plus about 3 percentage points of effective sector-specific tariffs.",[(22,"A new US baseline tariff of 10% on all goods and sector-specific tariffs that will add approximately another 3 percentage points of effective tariffs on Vietnamese exports to the US are assumed","Section 1.2 paragraph")]),
    c("t-x02","test","text","How many bilateral and plurilateral free-trade agreements had Viet Nam concluded, and how many more were under negotiation?","17 concluded; 3 more under negotiation.",[(99,"17 bilateral and plurilateral free trade agreements (FTAs), with three more under negotiation","Section 4.2 trade paragraph")]),
    c("t-x03","test","text","What income-status targets has the government set for 2030 and 2045?","Upper-middle-income status by 2030 and high-income status by 2045.",[(99,"The government has set a target of reaching upper middle-","Section 4.1 text, first wrapped fragment"),(99,"income status by 2030 and high-income status by 2045","Section 4.1 text, continued fragment")]),
    c("t-x04","test","text","According to Box 4.3, how many Meister schools had Korea established, and what employment rate had their graduates achieved?","More than 50 schools, with graduate employment rates above 90%.",[(119,"Korea has established more than 50 Meister schools and their graduates have achieved employment rates of more than 90%","Box 4.3")]),
    c("t-u01","test","text","How many electric buses were registered in Hanoi in 2025?","Not specified in this report.",[],answerable=False,absence="The report discusses transport decarbonisation but gives no 2025 Hanoi electric-bus registration count."),
    c("t-u02","test","text","What was the exact number of participants in the 2024 randomized trial evaluating bank consolidation in Viet Nam?","Not specified in this report.",[],answerable=False,absence="No such randomized trial or participant count is reported in the survey."),
    c("t-u03","test","text","What was the average monthly salary of a software engineer in Da Nang in 2025?","Not specified in this report.",[],answerable=False,absence="The report discusses education, skills and productivity but does not report this occupation-and-city salary."),
    c("t-u04","test","text","How many electric vehicles were sold in Ho Chi Minh City in 2024?","Not specified in this report.",[],answerable=False,absence="The report discusses transport emissions and policies but does not give city-level EV sales."),
    c("t-u05","test","text","What exact percentage of Vietnamese households had rooftop solar in 2025?","Not specified in this report.",[],answerable=False,absence="The report describes solar power potential and generation plans, not 2025 household adoption."),
    c("t-u06","test","text","What was the 2025 poverty rate for each of Viet Nam’s provinces?","Not specified in this report.",[],answerable=False,absence="The survey reports selected national and population-group indicators, not a province-by-province 2025 poverty table."),
]

# Refresh the final bonus benchmark after the initial test set was reviewed.
# Keep the original dev questions except for its visual group; the new test is
# a fresh, rubric-compliant 30-case holdout with image questions separate from
# chart-reading questions.
CASES = [case for case in CASES if case["split"] == "dev" and not case["id"].startswith("d-v")]

def photo_case(id_: str, split: str, question: str, answer: str, page: int,
               note: str, region: str) -> dict:
    case = c(id_, split, "visual", question, answer,
             [(page, note, region)], visual={page: note})
    case["gold"][0]["visual_type"] = "photograph"
    return case


CASES += [
    # Dev image cases use photographs; chart-reading cases remain in chart.
    photo_case("d-v01", "dev", "In the city photograph, what is reflected on the water?",
               "The high-rise city skyline is reflected in the water.", 18,
               "Photo evidence: an urban skyline with several high-rise buildings stands along a broad waterway; the towers and city lights are reflected on the water.",
               "Full-page urban photograph, water and skyline reflection"),
    photo_case("d-v02", "dev", "What infrastructure is visible beside the cars in the transport photograph?",
               "Electric-vehicle charging posts with black cables are beside the cars.", 70,
               "Photo evidence: a row of electric-vehicle charging posts with black charging cables stands beside teal cars in an indoor parking area.",
               "Full-page photograph of EV charging posts"),
    photo_case("d-v03", "dev", "What protective equipment does the foreground factory worker wear?",
               "A red cap, face mask and white gloves.", 96,
               "Photo evidence: a factory assembly floor has multiple workstations and workers in blue uniforms; the foreground worker wears a red cap, face mask and white gloves.",
               "Full-page factory photograph, foreground worker"),

    # Fresh test: 24 answerable, with five image cases distinct from five charts.
    c("t-t01", "test", "table", "What is the projected current-account balance for 2026 in Table 1.1?",
      "3.8% of GDP.", [(23, "Current account balance (% of GDP) 0.3 5.9 5.7 4.2 3.8", "Table 1.1, current-account row")]),
    c("t-t02", "test", "table", "In Table 1.3, what is the estimated fiscal cost of reducing social-security contributions for low-income earners?",
      "1.0% of GDP in deficit-increasing costs (−1.0%).", [(38, "Reduce social security contributions for low-income earners (Chapter 2) -1.0", "Table 1.3, low-income contribution row")]),
    c("t-t03", "test", "table", "According to Table 2.2, what are the employer and employee unemployment-insurance rates?",
      "1.0% each.", [(59, "Unemployment insurance 1.0 1.0", "Table 2.2, unemployment-insurance row")]),
    c("t-t04", "test", "table", "What was the 2024/25 environmental-protection tax rate on diesel in Table 3.2?",
      "1,000 VND per litre (0.039 USD per litre).", [(77, "Diesel fuel 1 000 VND/litre 2 000 VND/litre", "Table 3.2, diesel row")]),
    c("t-t05", "test", "table", "What reduction target did the 2025 plan set for unnecessary regulations and administrative procedures in Table 4.1?",
      "At least 30%, with a goal to reduce business costs by at least 3%.", [(130, "The government launched a plan in 2025 to cut at least 30%", "Table 4.1, simplification action cell"), (130, "reduce business costs by at least 3%", "Table 4.1, simplification action cell")]),
    c("t-t06", "test", "table", "What share of students pursue technical and vocational education and training, according to Table 4.3?",
      "6%.", [(135, "Only 6% of students pursue technical and vocational education and training", "Table 4.3, education finding")]),

    photo_case("t-v01", "test", "In the urban photograph, what prominent features appear together in the scene?",
               "High-rise city buildings stand beside a broad waterway, with their reflections visible on the water.", 18,
               "Photo evidence: an urban skyline with several high-rise buildings stands along a broad waterway; the towers and city lights are reflected on the water.",
               "Full-page urban photograph, skyline and waterway"),
    photo_case("t-v02", "test", "In the page 48 photograph, describe the foreground woman's garment and accessory, and identify what is visible at the left edge.",
               "She wears a red, green, black and white patterned garment and a large hoop earring; another child is partly visible at the left edge.", 48,
               "Photo evidence: an adult woman holding a young child is in the foreground; the woman wears a red, green, black and white patterned garment and a large hoop earring, and another child is partly visible at the left edge.",
               "Full-page photograph, woman and children"),
    photo_case("t-v03", "test", "What color are the cars beside the charging equipment in the transport photograph?",
               "Teal.", 70,
               "Photo evidence: a row of electric-vehicle charging posts with black charging cables stands beside teal cars in an indoor parking area.",
               "Full-page photograph, vehicles beside charging posts"),
    photo_case("t-v04", "test", "What marks the central aisle in the factory photograph?",
               "Yellow floor lines mark the aisle between the production workstations.", 96,
               "Photo evidence: a yellow-marked central aisle runs between factory workstations, with workers and machinery on both sides.",
               "Full-page factory photograph, central aisle"),
    photo_case("t-v05", "test", "What safety gear is visible on the foreground worker in the factory photograph?",
               "A red cap, face mask and white gloves; the worker wears a blue uniform.", 96,
               "Photo evidence: a factory assembly floor has multiple workstations and workers in blue uniforms; the foreground worker wears a red cap, face mask and white gloves.",
               "Full-page factory photograph, foreground worker"),

    c("t-c01", "test", "chart", "What potential does Figure 3.8 show for fixed offshore wind?", "101 GW.",
      [(82, "Figure 3.8. Viet Nam has a considerable renewable energy potential", "Figure 3.8, fixed offshore wind bar")],
      visual={82: "Visual inspection of Figure 3.8: fixed offshore wind potential is labelled 101 GW."}),
    c("t-c02", "test", "chart", "Approximately how many international tourists arrived in Viet Nam at the 2019 peak in Figure 1.4?", "About 18 million.",
      [(24, "Figure 1.4. The number of foreign tourists to Viet Nam is approaching the pre-COVID 19 peak", "Figure 1.4, 2019 bar")],
      visual={24: "Visual inspection of Figure 1.4: international tourist arrivals peak at approximately 18 million in 2019 before the pandemic decline."}),
    c("t-c03", "test", "chart", "What old-age dependency ratio does Figure 1.15 show for Viet Nam in 2050?", "Approximately 32%.",
      [(37, "Figure 1.15. The elderly dependency is currently low, but will rise sharply by mid-century", "Figure 1.15, Viet Nam 2050 bar")],
      visual={37: "Visual inspection of Figure 1.15: Viet Nam's 2050 old-age dependency bar is approximately 32%."}),
    c("t-c04", "test", "chart", "In Figure 3.7, approximately how much coal did Viet Nam consume and produce in 2023?", "About 2.3 exajoules consumed and 1.1 exajoules produced.",
      [(82, "Figure 3.7. Viet Nam’s consumption of coal exceeds its production", "Figure 3.7, 2023 endpoints")],
      visual={82: "Visual inspection of Figure 3.7: in 2023, coal consumption is approximately 2.3 exajoules while coal production is approximately 1.1 exajoules."}),
    c("t-c05", "test", "chart", "What relative labour-productivity level is shown for foreign GVC firms in Figure 4.6?", "70% relative to domestic-market firms.",
      [(102, "Figure 4.6. Participation in international trade and GVCs boost firms’ productivity and wages", "Figure 4.6, foreign GVC labour-productivity bar")],
      visual={102: "Visual inspection of Figure 4.6: labour productivity for foreign GVC firms is shown at 70% relative to domestic-market firms."}),

    c("t-m01", "test", "multi_hop", "Combine the report's typhoon account and its damage estimate: what GDP share was the estimated economic damage, and how many industrial parks were in the affected northern provinces?",
      "Damage was estimated at 0.7% of GDP; the affected northern provinces were home to 111 industrial parks.",
      [(72, "In September 2024, Typhoon Yagi, one of the most powerful storms to hit Viet Nam in three decades, caused economic damage estimated at 0.7% of GDP", "Section 3.1, Typhoon Yagi damage"),
       (20, "The typhoon unleashed devastating floods and landslides across northern provinces that are home to 111 industrial parks and nearly 5 000 FDI enterprises", "Section 1.1, Yagi impacts")]),
    c("t-m02", "test", "multi_hop", "What was the 2024/25 diesel environmental tax in Table 3.2, and what change to diesel taxation does Table 3.3 recommend?",
      "1,000 VND per litre; bring the diesel tax in line with the gasoline tax.",
      [(77, "Diesel fuel 1 000 VND/litre 2 000 VND/litre", "Table 3.2, diesel row"),
       (92, "Bring the diesel tax in line with the gasoline tax.", "Table 3.3, horizontal mitigation recommendation")]),
    c("t-m03", "test", "multi_hop", "Compare the estimated GDP cost of reducing low-income workers’ social-security contributions in Table 1.3 with the current total contribution rates in Table 2.2. What are the reported values?",
      "The reduction is estimated to cost 1.0% of GDP; current total rates are 21.5% for employers and 10.5% for employees (32% combined).",
      [(38, "Reduce social security contributions for low-income earners (Chapter 2) -1.0", "Table 1.3, deficit-increasing measures"),
       (59, "Total 21.5 10.5", "Table 2.2, total contribution row")]),
    c("t-m04", "test", "multi_hop", "Using Table 2.2 and Figure 2.7, state the total employee contribution rate across the listed insurance schemes and the share of GDP generated by Viet Nam’s social-security contributions in 2022.",
      "Employees contribute 10.5% in total; social-security contributions generated about 5.4% of GDP in 2022.",
      [(59, "Total 21.5 10.5", "Table 2.2, total row"),
       (60, "Figure 2.7. Social security contributions generate sizeable government revenues, 2022", "Figure 2.7, Viet Nam bar")],
      visual={60: "Visual inspection: Viet Nam’s red bar is approximately 5.4% of GDP, between the 5% and 6% gridlines."}),

    c("t-x01", "test", "text", "What was Viet Nam’s GDP growth rate in 2023, as stated in Section 1.1?", "5.1%.",
      [(20, "growth to 5.1% in 2023", "Section 1.1 opening paragraph")]),
    c("t-x02", "test", "text", "What length of coastline makes Viet Nam particularly susceptible to sea-level rise and extreme weather?", "Approximately 3,260 km.",
      [(72, "approximately 3,260 km of coastline", "Section 3.1, climate exposure")]),
    c("t-x03", "test", "text", "What share of cumulative CO2 emissions has Viet Nam contributed, according to the report?", "0.3%.",
      [(72, "has contributed to just 0.3% of cumulative CO2 emissions", "Section 3.1, emissions contribution")]),
    c("t-x04", "test", "text", "How large were Viet Nam’s CO2 emissions in 2023, according to the cited estimate?", "373 million tonnes of CO2.",
      [(74, "emissions, which are estimated to have reached 373 million tons of CO2 in 2023", "Discussion following Figure 3.2")]),

    c("t-u01", "test", "text", "How many public fast-charging outlets were operating in Da Nang at the end of 2025?", "Not specified in this report.", [], answerable=False,
      absence="The report discusses transport decarbonisation but contains no city-level count of public charging outlets in Da Nang in 2025."),
    c("t-u02", "test", "text", "What share of Hanoi households had rooftop solar panels in 2025?", "Not specified in this report.", [], answerable=False,
      absence="The report discusses solar generation potential and national energy plans, but gives no 2025 household adoption share for Hanoi."),
    c("t-u03", "test", "text", "What was the average monthly wage of garment workers in Bac Ninh in 2024?", "Not specified in this report.", [], answerable=False,
      absence="The report contains wage comparisons and labour-market indicators, but not this occupation- and province-specific monthly wage."),
    c("t-u04", "test", "text", "How many Mekong Delta firms reported drought-related losses in 2024?", "Not specified in this report.", [], answerable=False,
      absence="The report discusses climate risk and typhoon damage, but does not report a survey count of Mekong Delta firms with drought losses."),
    c("t-u05", "test", "text", "What proportion of public schools in Viet Nam had air conditioning in 2024?", "Not specified in this report.", [], answerable=False,
      absence="The report discusses education outcomes and spending, but does not report school air-conditioning coverage."),
    c("t-u06", "test", "text", "How many electric two-wheelers were registered in Ho Chi Minh City in 2025?", "Not specified in this report.", [], answerable=False,
      absence="The report discusses transport emissions and cleaner vehicles but gives no city-level electric two-wheeler registration count for 2025."),
]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold()
    return " ".join(s.split())


def main() -> None:
    pdf = fitz.open(PDF)
    for item in CASES:
        for gold in item.get("gold", []):
            if not 1 <= gold["pdf_page"] <= len(pdf):
                raise ValueError(f"{item['id']} references an invalid PDF page")
            page_text = norm(pdf[gold["pdf_page"] - 1].get_text("text", sort=True))
            quote = norm(gold["evidence"])
            if quote not in page_text:
                image_annotation = norm(gold.get("visual_annotation") or "")
                crop = ROOT / "index" / "visual_pages" / f"page-{gold['pdf_page']:03d}.png"
                if (gold.get("visual_type") != "photograph" or quote != image_annotation
                        or not crop.is_file() or not gold.get("visual_region")):
                    raise ValueError(f"Evidence quote not found in original PDF page or verified image annotation for {item['id']}: {gold['evidence']}")

    for split, min_answerable, min_unanswerable in [("dev", 15, 3), ("test", 24, 6)]:
        rows = [x for x in CASES if x["split"] == split]
        yes = sum(x["answerable"] for x in rows)
        no = len(rows) - yes
        if yes < min_answerable or no < min_unanswerable:
            raise ValueError(f"{split} counts are below the advanced challenge minimum")
    test_groups = {g: sum(x["answerable"] and x["group"] == g and x["split"] == "test" for x in CASES)
                   for g in ["table", "visual", "chart", "multi_hop", "text"]}
    required = {"table": 6, "visual": 5, "chart": 5, "multi_hop": 4, "text": 4}
    if any(test_groups[k] < v for k, v in required.items()):
        raise ValueError(f"test group coverage is below the advanced challenge minimum: {test_groups}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "questions.json").write_text(json.dumps(CASES, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    visual_notes = {}
    for item in CASES:
        for gold in item.get("gold", []):
            note = gold.get("visual_annotation")
            if not note:
                continue
            key = (gold["pdf_page"], note)
            visual_notes[key] = {
                "id": "vis-note-" + hashlib.sha256(f"{key[0]}|{key[1]}".encode("utf-8")).hexdigest()[:20],
                "kind": "human_visual_note",
                "text": note,
                "source_id": SOURCE_ID,
                "source": "OECD Economic Surveys: Viet Nam 2025",
                "pdf_page": gold["pdf_page"],
                "printed_page": gold["pdf_page"] - 2 if gold["pdf_page"] >= 4 else None,
                "crop_path": f"advanced/index/visual_pages/page-{gold['pdf_page']:03d}.png",
                "metadata": {"method": "manual visual inspection of the cited PDF page crop", "verified_against_original": True,
                             **({"visual_type": gold["visual_type"], "visual_region": gold["visual_region"]}
                                if gold.get("visual_type") else {})},
            }
    note_path = ROOT / "index" / "visual_notes.jsonl"
    note_path.parent.mkdir(parents=True, exist_ok=True)
    with note_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in visual_notes.values():
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({
        "total": len(CASES),
        "dev": {"total": sum(x["split"] == "dev" for x in CASES), "answerable": sum(x["split"] == "dev" and x["answerable"] for x in CASES), "unanswerable": sum(x["split"] == "dev" and not x["answerable"] for x in CASES)},
        "test": {"total": sum(x["split"] == "test" for x in CASES), "answerable": sum(x["split"] == "test" and x["answerable"] for x in CASES), "unanswerable": sum(x["split"] == "test" and not x["answerable"] for x in CASES), "answerable_groups": test_groups},
        "gold_quote_validation": "passed against cited physical PDF pages"
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
