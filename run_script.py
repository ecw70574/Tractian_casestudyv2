from linkdapi import LinkdAPI
import requests
import os
import time 
import pycountry 
from collections import OrderedDict
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import re


provided_companies = {
    "Nike": "https://www.nike.com",
    "Boeing": "https://www.boeing.com",
    "Databricks": "https://www.databricks.com",
    "Hulu": "https://www.hulu.com",
    "PepsiCo": "https://www.pepsico.com",
    "Chipotle": "https://www.chipotle.com",
    "Olive Garden": "https://www.olivegarden.com",
    "Chevron": "https://www.chevron.com",
    "United Airlines": "https://www.united.com",
    "Georgia-Pacific": "https://www.gp.com",
    "Lululemon": "https://www.lululemon.com",
    "Ikea": "https://www.ikea.com",
    "Walmart": "https://www.walmart.com",
    "Starbucks": "https://www.starbucks.com",
    "Intel": "https://www.intel.com",
    "Dell": "https://www.dell.com",
    "Tesla": "https://www.tesla.com", 
    "Under Armour": "https://www.underarmour.com",
    "Chanel": "https://www.chanel.com"
}

def company_name_lookup(company_name: str):
    linkdapi_key = os.getenv("LINKDAPI_KEY")
    client = LinkdAPI(linkdapi_key) # initialize LinkdAPI client with API key
    url = "https://linkdapi.com/api/v1/companies/name-lookup?query=" + company_name
    headers = {
        "X-linkdapi-apikey": linkdapi_key
    }
    response = requests.get(url, headers=headers)
    # store JSON dict
    data = response.json()
    companies = data.get("data", {}).get("companies", [])
    if not companies:
        print(f"No matches found for '{company_name}'")
        return None
    # Get first result to avoid fuzzy matching issue (different company aliases)
    # Making assumption that top result is most relevant 
    first_hit = companies[0]
    return first_hit

def fetch_company_info(company_name: str):
    linkdapi_key = os.getenv("LINKDAPI_KEY")
    first_hit = company_name_lookup(company_name)
    if not first_hit:
        return None
    company_id = first_hit.get("id")
    official_name = first_hit.get("displayName") # official name as param
    if not company_id:
        print(f"No company ID found for '{company_name}'")
        return None
    url2 = f"https://linkdapi.com/api/v1/companies/company/info?id={company_id}&name={official_name}"
    headers = {
        "X-linkdapi-apikey": linkdapi_key
    }
    response2 = requests.get(url2, headers=headers) # make next api call to get company details
    return response2.json() # return company details as JSON dict


# Need to Geocode because don't have exact addresses
# Nomatim OpenStreetMap API for geocoding
# need to make a query with fields we have
def build_nominatim_query(addr, company=None, include_company_name = False, include_postal=False): # VSCode/Github Copilot suggested function logic
    # Map country code to full name if possible
    country_code = addr.get("country") or addr.get("countryCode")
    country = None
    if country_code:
        try:
            country_obj = pycountry.countries.get(alpha_2=country_code.upper())
            country = country_obj.name if country_obj else country_code
        except:
            country = country_code

    # Build parts in order of relevance to Nomatim
    parts = [
        company if include_company_name else None,
        addr.get("city"),
        addr.get("geographicArea")
    ]
    print(addr.get("line1"))
    if include_postal:
        parts.append(addr.get("postalCode"))

    parts.append(country)

    # Remove empty parts and duplicates (keep order)
    parts = [p for p in parts if p]
    parts = list(OrderedDict.fromkeys(parts))

    # Join into a single string
    query = ", ".join(parts)

    return query


# deterministic scoring function
def score_company_deterministic(company_details):
    """
    Score a company on 6 features using its structured fields:
    - description
    - industriesV2
    - industriesLegacy
    - specialties
    - staffCount
    Returns a dictionary with scores between 0 and 1.
    """

    # Safely default missing fields
    company_details.setdefault("description", "")
    company_details.setdefault("industriesV2", [])
    company_details.setdefault("industriesLegacy", [])
    company_details.setdefault("specialties", [])
    company_details.setdefault("staffCount", 0)

    # Extract fields safely
    desc = company_details.get("description", "").lower()
    industries = [i.lower() for i in company_details.get("industriesV2", [])]
    parent_industries = [i.lower() for i in company_details.get("industriesLegacy", [])]
    specialties = [s.lower() for s in company_details.get("specialties", [])]
    staff_count = company_details.get("staffCount", 0)

    # Feature scoring logic
    # Industrial sector fit (does it match key ICP industries?)
    # Use semantic embeddings for similarity rather than keyword matching
    model = SentenceTransformer("all-MiniLM-L6-v2")
    company_text = " ".join(industries + parent_industries)
    icp_texts = ["industrial manufacturing", "heavy machinery", "mechanical components", "energy logistics", "mining", "automotive", "chemical", "materials", "car manufacturing"]
    emb_company = model.encode([company_text])
    emb_icp = model.encode(icp_texts)
    scores = np.dot(emb_company, emb_icp.T) / (np.linalg.norm(emb_company) * np.linalg.norm(emb_icp, axis=1))
    industrial_sector_fit = float(np.max(scores))  # 0–1

    # Equipment intensity (keywords in description / specialties)
    equipment_keywords = ["machine", "equipment", "motor", "pump", "compressor", "gearbox", "conveyor", "rotating", "bearing", "vibration", "rpm"]
    equipment_intensity = min(1.0, 0.2 + 0.16 * sum(kw in desc or kw in specialties for kw in equipment_keywords))

    # Maintenance operations (keywords indicating maintenance focus)
    maintenance_keywords = ["maintenance", "monitoring", "predictive", "downtime", "inspection", "ai-assisted", "condition-based", "optimization", "insights", "reliability", "asset management", "shop floor", "technician"]
    maintenance_operations = min(1.0, 0.2 + 0.16 * sum(kw in desc or kw in specialties for kw in maintenance_keywords))

    # Operational footprint (number of locations implied in description or staff size)
    operational_footprint = min(1.0, staff_count / 2000.0)  # scale up to 2000 staff

    # Cost of downtime (assume high if heavy machinery / industrial)
    cost_of_downtime = 0.8 if industrial_sector_fit > 0.5 and equipment_intensity > 0.5 else 0.3

    # Company size (staff count scale)
    company_size = min(1.0, staff_count / 2000.0)  # same scale as footprint

    # Return JSON
    scores = {
        "industrial_sector_fit": round(industrial_sector_fit, 2),
        "equipment_intensity": round(equipment_intensity, 2),
        "maintenance_operations": round(maintenance_operations, 2),
        "operational_footprint": round(operational_footprint, 2),
        "cost_of_downtime": round(cost_of_downtime, 2),
        "company_size": round(company_size, 2)
    }
    # return the ICP field that had the highest score 
    max_icp_feature_score = icp_texts[np.argmax(scores)]
    return scores, max_icp_feature_score # return max score category for explainability

def score_company_total_10(company_details, weights=None):
    """
    Returns a single total score scaled to 10
    """
    # Compute individual 0–1 scores
    scores, max_icp_feature_score = score_company_deterministic(company_details)

    # Default: equal weighting if not provided
    if weights is None:
        weights = {
            "industrial_sector_fit": 2,
            "equipment_intensity": 2,
            "maintenance_operations": 1.5,
            "operational_footprint": 1,
            "cost_of_downtime": 1,
            "company_size": 1
        }

    # Weighted sum
    weighted_sum = sum(scores[f] * w for f, w in weights.items())
    max_possible = sum(weights.values())

    # Scale to 0–10
    total_score = round((weighted_sum / max_possible) * 10, 0)
    return total_score, max_icp_feature_score

def geocode_location(company_details, company_name):
    geoapify_key = os.getenv("GEOAPIFY_KEY")
    addresses = [] 
    address_types = []

    # Try different Nominatim Queries to see which gets a hit for Geocoding
    location_dict = company_details.get("locations", [])
    print(f"Locations found for '{company_name}': {location_dict}")
    for loc in location_dict:
        # for formatting in table
        addresses.append(build_nominatim_query(loc, company=company_name, include_company_name=False, include_postal=False))
        print("entered loc loop")

        # First try company-level query
        query_company = build_nominatim_query(loc, company=company_name, include_company_name=True)
        headers_nom = {"User-Agent": "tractian-case-study"}
        response3 = requests.get(f"https://nominatim.openstreetmap.org/search?q={query_company}&format=json&limit=1", headers=headers_nom)
        data3 = response3.json()

        if data3:
            print(f"Successful Nominatim geocoding with query: '{query_company}'")
            place_type = data3[0]["type"]
            print(f"Place type from Nominatim: '{place_type}'")
            address_types.append(place_type)
        elif loc.get("line1"):  # Then try street-level query
            query_street = build_nominatim_query(loc, include_company_name=False)
            response3 = requests.get(f"https://nominatim.openstreetmap.org/search?q={query_street}&format=json&limit=1", headers=headers_nom)
            data3 = response3.json()
            if data3:
                print(f"Successful Nominatim geocoding with street query: '{query_street}'")
                place_type = data3[0]["type"]
                print(f"Place type from Nominatim with street query: '{place_type}'")
                address_types.append(place_type)
            else:  # GeoApify fallback
                url = "https://api.geoapify.com/v1/geocode/search"
                params = {"text": query_company, "apiKey": geoapify_key}
                resp = requests.get(url, params=params, headers={"Accept": "application/json"})
                data4 = resp.json()
                if data4.get("features"):
                    formatted_address = data4["features"][0]["properties"].get("formatted", "unknown")
                    print(f"Successful GeoApify geocoding: '{formatted_address}'")
                    address_types.append(formatted_address)
                else:
                    address_types.append("unknown")
        else:
            address_types.append("unknown")

        time.sleep(1) # to avoid hitting rate limits on APIs

    return addresses, address_types

# Prepare DataFrames
master_df = pd.DataFrame(columns=['Company', 'Website', 'Facility Location', "Facility Type", "ICP Fit Score", "Primary ICP Feature"])
mini_df = pd.DataFrame(columns=['Company', 'Website', 'Facility Location', "Facility Type", "ICP Fit Score", "Primary ICP Feature"])

# Main script logic
for company_name, company_website in provided_companies.items():
    # Provided
    # Get Company Details from LinkdAPI (2 endpoints needed - first to get company ID, second to get details)
    company_info = fetch_company_info(company_name)
    if not company_info or "data" not in company_info:
        print(f"No details found for {company_name}, skipping.")
        continue
    company_details = company_info.get("data", {})

    print(company_details)
    # Extract company-level features and score on ICP fit using deterministic function
    # Extract location details and use for later geocoding
    # Deterministic scoring using keywords and semantic similarity for ICP fit, 6 priority features each get a score and relative weighting 
    score_10, max_icp_feature_score = score_company_total_10(company_details)
    print("Company " + company_name + " scores " + str(score_10))

    # Geocode using Nominatim and GeoApify to get facility locations and types (HQ, branch, etc.) for table display 
    # Try different query
    addresses, address_types = geocode_location(company_details, company_name)

    length = len(address_types)
    # make into a temp dataframe
    mini_df = pd.DataFrame({
        "Company": [company_name]*length,
        "Website": [company_website]*length,
        "Facility Location": addresses[:length],
        "Facility Type": address_types,
        "ICP Fit Score": [score_10]*length,
        "Primary ICP Feature": [max_icp_feature_score]*length
    })

    master_df = pd.concat([master_df, mini_df], ignore_index=True)

# Save results
master_df.to_csv("company_scoring_results.csv", index=False)