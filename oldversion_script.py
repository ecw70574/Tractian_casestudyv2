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
import os

csv_path = "company_scoring_results.csv"
# remove old CSV if exists
if os.path.exists(csv_path):
    os.remove(csv_path)


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
    # check to see if any hits (company list not empty)

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

    # Extract fields safely
    desc = company_details.get("description", "").lower()
    industries = [i.lower() for i in company_details.get("industriesV2", [])]
    parent_industries = [i.lower() for i in company_details.get("industriesLegacy", [])]
    specialties = [s.lower() for s in company_details.get("specialties", [])]
    staff_count = company_details.get("staffCount", 0)
      # default 1 if not provided


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

    # icp_industries = ["industrial", "manufacturing", "heavy machinery", "mechanical", "mining", "energy logistics", "chemical", "automotive", "materials", "car"]
    # industrial_sector_fit = 1.0 if any(ind in icp_industries for ind in industries + parent_industries) else 0.2

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
    ifHQ = False
    # company_details = fetch_company_info(og_company_name).get("data", {})
    type(company_details)
    location_dict = company_details["locations"]
    print(f"Locations found for '{company_name}': {location_dict}")
    for loc in location_dict:
        # for formatting in table
        addresses.append(build_nominatim_query(loc, company=company_name, include_company_name=False, include_postal = False))
        print("entered loc loop")
        # ifHQ = loc.get("headquarter")
        if loc.get("headquarter"):
            address_types.append("HQ")
        else:
            query=build_nominatim_query(loc, company=company_name, include_company_name=True, include_postal = True) # original name provided
            # queries.append(build_nominatim_query(loc, company=og_company_name, include_company_name=True, include_postal = True)) # add postal, sometimes confuses Nomatim
            url3 = "https://nominatim.openstreetmap.org/search?" + "q=" + query + "&format=json&limit=1"
            headers_nom = {
                "User-Agent": "tractian-case-study"  
            }
            response3 = requests.get(url3, headers = headers_nom) # No API key needed
            data3 = response3.json()
            if data3:
                print(f"Successful Nominatim geocoding with query: '{query}'")
                place_type = data3[0]["type"]
                print(f"Place type from Nominatim: '{place_type}'")
                address_types.append(place_type)
            else:
                print(f"No Nomatim geocoding result for query: '{query}'")
                
                # Try another strategy - GeoApify API 
                url = "https://api.geoapify.com/v1/geocode/search"
                params = {
                    "text": query,
                    "apiKey": geoapify_key
                }
                headers_geoap = {"Accept": "application/json"}

                resp = requests.get(url, params=params, headers=headers_geoap)
                data4 = resp.json()
                if data4:
                    print(f"Successful GeoApify geocoding with query: '{query}'")
                    formatted_address = data4["features"][0]["properties"].get("formatted")
                    print(formatted_address)
                    
                    # Use formatted address to try Nominatim again
                    url3 = "https://nominatim.openstreetmap.org/search?" + "q=" + formatted_address + "&format=json&limit=1"
                    response3 = requests.get(url3, headers = headers_nom)
                    data3 = response3.json()
                    if data3:
                        print(f"Successful Nominatim geocoding with GeoApify formatted address: '{formatted_address}'")
                        place_type = data3[0]["type"]
                        address_types.append(place_type)
                        print(f"Place type from Nominatim with GeoApify address: '{place_type}'")
                    else:
                        print(f"No Nomatim geocoding result for GeoApify formatted address: '{formatted_address}'")

                else:
                    print(f"No GeoApify geocoding result for query: '{query}'")
        time.sleep(1) # to avoid hitting rate limits on APIs
        # If after all attempts we still have no type for this location, mark as Unknown
        if len(address_types) < len(addresses):
            address_types.append("Unknown")
            print(f"No geocoding result for '{company_name}' location after all attempts; marked as 'Unknown'")
    return addresses, address_types
                
master_df = pd.DataFrame(columns=['Company', 'Website', 'Facility Location', "Facility Type", "ICP Fit Score", "Primary ICP Feature"])
mini_df = pd.DataFrame(columns=['Company', 'Website', 'Facility Location', "Facility Type", "ICP Fit Score", "Primary ICP Feature"])

# Main script logic
for company_name, company_website in provided_companies.items():
    # Provided
    # Get Company Details from LinkdAPI (2 endpoints needed - first to get company ID, second to get details)
    # company_details = fetch_company_info(company_name).get("data", {})
    company_details = fetch_company_info(company_name)
    if not company_info:
        print(f"Can't find company '{company_name}'. Moving to the next.")
        continue  # skips this company but keeps the loop running
    print(company_details)
    # Extract company-level features and score on ICP fit using deterministic function
    # Extract location details and use for later geocoding
    # Deterministic scoring using keywords and semantic similarity for ICP fit, 6 priority features each get a score and relative weighting 
    score_10, max_icp_feature_score = score_company_total_10(company_details)
    print(score_10)

    # Geocode using Nominatim and GeoApify to get facility locations and types (HQ, branch, etc.) for table display 
    # Try different query
    addresses, address_types = geocode_location(company_details, company_name)

    length = len(address_types)
    # make into a temp dataframe

    mini_df["Facility Location"] = addresses[0:length]
    mini_df["Facility Type"] = address_types
    mini_df["Company"] = company_name
    mini_df["Website"] = company_website
    mini_df["ICP Fit Score"] = score_10
    mini_df["Primary ICP Feature"] = max_icp_feature_score

    master_df = pd.concat([master_df, mini_df], ignore_index=True)
    mini_df.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path)) # save csv after each company in case of crash
    mini_df = mini_df.iloc[0:0] # drop all rows to reset for next company

    # Essentially sends in batches of 1 company at a time


pd.to_csv(master_df, "company_scoring_results.csv", index=False)
