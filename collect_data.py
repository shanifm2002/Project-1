import os
import pandas as pd
import wbdata
import yfinance as yf
from datetime import datetime
from tqdm import tqdm

# ----------------------------
# CONFIGURATION
# ----------------------------
start_year = 2010
end_year = 2025

# ----------------------------
# COUNTRY LIST (ISO-2 codes)
# ----------------------------
countries = {
    # 🌍 Africa
    'DZ': 'Algeria', 'AO': 'Angola', 'BJ': 'Benin', 'BW': 'Botswana', 'BF': 'Burkina Faso',
    'BI': 'Burundi', 'CV': 'Cabo Verde', 'CM': 'Cameroon', 'CF': 'Central African Republic',
    'TD': 'Chad', 'KM': 'Comoros', 'CG': 'Congo (Brazzaville)', 'CD': 'Congo (Kinshasa)',
    'DJ': 'Djibouti', 'EG': 'Egypt', 'GQ': 'Equatorial Guinea', 'ER': 'Eritrea',
    'SZ': 'Eswatini', 'ET': 'Ethiopia', 'GA': 'Gabon', 'GM': 'Gambia', 'GH': 'Ghana',
    'GN': 'Guinea', 'GW': 'Guinea-Bissau', 'CI': 'Ivory Coast', 'KE': 'Kenya', 'LS': 'Lesotho',
    'LR': 'Liberia', 'LY': 'Libya', 'MG': 'Madagascar', 'MW': 'Malawi', 'ML': 'Mali',
    'MR': 'Mauritania', 'MU': 'Mauritius', 'MA': 'Morocco', 'MZ': 'Mozambique', 'NA': 'Namibia',
    'NE': 'Niger', 'NG': 'Nigeria', 'RW': 'Rwanda', 'ST': 'Sao Tome & Principe', 'SN': 'Senegal',
    'SC': 'Seychelles', 'SL': 'Sierra Leone', 'SO': 'Somalia', 'ZA': 'South Africa',
    'SS': 'South Sudan', 'SD': 'Sudan', 'TZ': 'Tanzania', 'TG': 'Togo', 'TN': 'Tunisia',
    'UG': 'Uganda', 'ZM': 'Zambia', 'ZW': 'Zimbabwe',

    # 🌏 Asia
    'AF': 'Afghanistan', 'AM': 'Armenia', 'AZ': 'Azerbaijan', 'BH': 'Bahrain', 'BD': 'Bangladesh',
    'BT': 'Bhutan', 'BN': 'Brunei', 'KH': 'Cambodia', 'CN': 'China', 'CY': 'Cyprus',
    'GE': 'Georgia', 'IN': 'India', 'ID': 'Indonesia', 'IR': 'Iran', 'IQ': 'Iraq',
    'IL': 'Israel', 'JP': 'Japan', 'JO': 'Jordan', 'KZ': 'Kazakhstan', 'KW': 'Kuwait',
    'KG': 'Kyrgyzstan', 'LA': 'Laos', 'LB': 'Lebanon', 'MY': 'Malaysia', 'MV': 'Maldives',
    'MN': 'Mongolia', 'MM': 'Myanmar', 'NP': 'Nepal', 'KP': 'North Korea', 'OM': 'Oman',
    'PK': 'Pakistan', 'PS': 'Palestine', 'PH': 'Philippines', 'QA': 'Qatar', 'SA': 'Saudi Arabia',
    'SG': 'Singapore', 'KR': 'South Korea', 'LK': 'Sri Lanka', 'SY': 'Syria', 'TW': 'Taiwan',
    'TJ': 'Tajikistan', 'TH': 'Thailand', 'TL': 'Timor-Leste', 'TM': 'Turkmenistan',
    'AE': 'UAE', 'UZ': 'Uzbekistan', 'VN': 'Vietnam', 'YE': 'Yemen',

    # 🇪🇺 Europe
    'AL': 'Albania', 'AD': 'Andorra', 'AT': 'Austria', 'BY': 'Belarus', 'BE': 'Belgium',
    'BA': 'Bosnia & Herzegovina', 'BG': 'Bulgaria', 'HR': 'Croatia', 'CZ': 'Czech Republic',
    'DK': 'Denmark', 'EE': 'Estonia', 'FI': 'Finland', 'FR': 'France', 'DE': 'Germany',
    'GR': 'Greece', 'HU': 'Hungary', 'IS': 'Iceland', 'IE': 'Ireland', 'IT': 'Italy',
    'XK': 'Kosovo', 'LV': 'Latvia', 'LI': 'Liechtenstein', 'LT': 'Lithuania', 'LU': 'Luxembourg',
    'MT': 'Malta', 'MD': 'Moldova', 'MC': 'Monaco', 'ME': 'Montenegro', 'NL': 'Netherlands',
    'MK': 'North Macedonia', 'NO': 'Norway', 'PL': 'Poland', 'PT': 'Portugal', 'RO': 'Romania',
    'RU': 'Russia', 'SM': 'San Marino', 'RS': 'Serbia', 'SK': 'Slovakia', 'SI': 'Slovenia',
    'ES': 'Spain', 'SE': 'Sweden', 'CH': 'Switzerland', 'UA': 'Ukraine', 'GB': 'United Kingdom',

    # 🌎 North America
    'AG': 'Antigua & Barbuda', 'BS': 'Bahamas', 'BB': 'Barbados', 'BZ': 'Belize',
    'CA': 'Canada', 'CR': 'Costa Rica', 'CU': 'Cuba', 'DM': 'Dominica',
    'DO': 'Dominican Republic', 'SV': 'El Salvador', 'GD': 'Grenada', 'GT': 'Guatemala',
    'HT': 'Haiti', 'HN': 'Honduras', 'JM': 'Jamaica', 'MX': 'Mexico', 'NI': 'Nicaragua',
    'PA': 'Panama', 'KN': 'Saint Kitts & Nevis', 'LC': 'Saint Lucia',
    'VC': 'Saint Vincent & Grenadines', 'TT': 'Trinidad & Tobago', 'US': 'USA',

    # 🇧🇷 South America
    'AR': 'Argentina', 'BO': 'Bolivia', 'BR': 'Brazil', 'CL': 'Chile', 'CO': 'Colombia',
    'EC': 'Ecuador', 'GY': 'Guyana', 'PY': 'Paraguay', 'PE': 'Peru', 'SR': 'Suriname',
    'UY': 'Uruguay', 'VE': 'Venezuela',

    # 🇦🇺 Oceania
    'AU': 'Australia', 'FJ': 'Fiji', 'KI': 'Kiribati', 'MH': 'Marshall Islands',
    'FM': 'Micronesia', 'NR': 'Nauru', 'NZ': 'New Zealand', 'PW': 'Palau',
    'PG': 'Papua New Guinea', 'WS': 'Samoa', 'SB': 'Solomon Islands', 'TO': 'Tonga',
    'TV': 'Tuvalu', 'VU': 'Vanuatu'
}

# ----------------------------
# World Bank indicators
# ----------------------------
indicators = {
    'FP.CPI.TOTL.ZG': 'Inflation_Rate',    # Inflation (%)
    'NY.GDP.MKTP.KD.ZG': 'GDP_Growth',     # GDP Growth (%)
    'SL.UEM.TOTL.ZS': 'Unemployment'       # Unemployment (%)
}

# ----------------------------
# STEP 1: GLOBAL FINANCIAL DATA
# ----------------------------
print("📡 Fetching global market indicators (Oil Price, USD Index)...")

tickers = {
    'CL=F': 'Oil_Price',      # WTI Crude Oil
    'DX-Y.NYB': 'USD_Index'   # U.S. Dollar Index
}

yf_data = pd.DataFrame()
for symbol, name in tickers.items():
    data = yf.download(symbol, start=f"{start_year}-01-01", end=f"{end_year}-01-01", interval="1mo")['Close']
    data = data.resample('Y').mean()  # Convert monthly to yearly average
    yf_data[name] = data

yf_data.reset_index(inplace=True)
yf_data.rename(columns={'Date': 'Year'}, inplace=True)
yf_data['Year'] = yf_data['Year'].dt.year

# ----------------------------
# STEP 2: WORLD BANK COUNTRY DATA
# ----------------------------
print("\n🌍 Collecting yearly inflation, GDP, and unemployment data from World Bank...")
all_data = []

for code, name in tqdm(countries.items()):
    try:
        df = wbdata.get_dataframe(indicators, country=code)
        df.reset_index(inplace=True)
        df.rename(columns={'country': 'Country', 'date': 'Year'}, inplace=True)
        df['Year'] = df['Year'].astype(int)
        df['Country'] = name
        all_data.append(df)
    except Exception as e:
        print(f"⚠ Skipping {name} ({code}) due to error: {e}")

# Combine all countries
if all_data:
    combined = pd.concat(all_data, ignore_index=True)
else:
    print("❌ No data to concatenate. Exiting script.")
    combined = None

# ----------------------------
# STEP 3: MERGE WITH GLOBAL MARKET DATA
# ----------------------------
if combined is not None:
    print("\n🔄 Merging World Bank data with oil and USD index...")
    merged = pd.merge(combined, yf_data, on='Year', how='left')

    # Keep only relevant years (2010 onwards)
    merged = merged[merged['Year'].between(start_year, end_year)]
else:
    merged = None

# ----------------------------
# STEP 4: CLEAN & EXPORT
# ----------------------------
if merged is not None:
    merged = merged.round(2)
    merged = merged.dropna(subset=['Inflation_Rate'])
    merged.sort_values(['Country', 'Year'], inplace=True)

    # Reorder columns
    merged = merged[['Country', 'Year', 'Inflation_Rate', 'GDP_Growth', 'Unemployment', 'Oil_Price', 'USD_Index']]

    # Create data folder if not exists
    os.makedirs("data", exist_ok=True)

    # Save to CSV file inside /data folder
    output_file = os.path.join("data", "inflation_yearly_dataset.csv")
    merged.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n✅ Clean dataset saved successfully to: {output_file}")
    print(f"🌎 Total countries included: {len(countries)}")
    print("\n📊 Preview of the dataset:")
    print(merged.head(10))
else:
    print("❌ No merged data to clean or export.")
