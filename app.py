from shiny import App, ui, render, reactive
import folium
from geopy.geocoders import Nominatim
import joblib
import pandas as pd

# Load the saved model
model = joblib.load("xgb_pipeline_model.pkl")

# List of required features
NUMERICAL_FEATURES = [
    'taxAssessedValue', 'nearbyHomes/2/price', 'resoFacts/bathroomsFull',
    'mortgageRates/thirtyYearFixedRate', 'bathrooms', 'adTargets/sqft',
    'priceHistory/4/price', 'adTargets/bd', 'schools/1/rating',
    'resoFacts/bathroomsHalf', 'resoFacts/fireplaces', 'photoCount',
    'nearbyHomes/2/bathrooms', 'nearbyHomes/1/bathrooms', 'nearbyHomes/0/bathrooms',
    'nearbyHomes/2/livingArea', 'restimateHighPercent', 'nearbyHomes/1/livingArea',
    'ssid', 'nearbyHomes/0/livingArea', 'nearbyHomes/1/bedrooms',
    'nearbyHomes/2/bedrooms', 'resoFacts/onMarketDate', 'priceHistory/1/time',
    'resoFacts/atAGlanceFacts/1/factValue'
]
CATEGORICAL_FEATURES = [
    'adTargets/prange', 'attributionInfo/lastChecked', 'resoFacts/listingTerms',
    'resoFacts/atAGlanceFacts/3/factValue', 'resoFacts/specialListingConditions',
    'adTargets/zestibuck', 'adTargets/sqftrange', 'priceHistory/1/date',
    'nearbyHomes/1/listingTypeDimension', 'adTargets/price_band',
    'priceHistory/1/attributeSource/infoString3', 'resoFacts/highSchoolDistrict',
    'resoFacts/appliances/0', 'nearbyHomes/0/listingTypeDimension', 'resoFacts/lotSize',
    'isPremierBuilder', 'resoFacts/propertySubType/0', 'nearbyHomes/1/homeStatus',
    'priceHistory/2/date', 'nearbyHomes/1/hdpTypeDimension',
    'resoFacts/interiorFeatures/0', 'resoFacts/isNewConstruction',
    'resoFacts/waterSource/0', 'nearbyHomes/1/address/city',
    'resoFacts/fireplaceFeatures/0'
]

# User Interface
app_ui = ui.page_fluid(
    ui.h2("Real Estate Price Prediction Tool", style="background-color: #6180c2; color: white; padding: 10px; border-radius: 5px;"),
    ui.row(
        # Left column with inputs
        ui.column(
            6,
            ui.div(
                ui.h3("Input", style="background-color: #6180c2; color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: 0;"),
                ui.div(
                    ui.input_text("address_input", "Enter Address", ""),
                    ui.input_text("city", "Enter City", ""),
                    ui.input_text("state", "Enter State", ""),
                    ui.input_text("zip_code", "Enter Zipcode", ""),
                    ui.input_text_area("num_bedroom", "Enter Number of Bedrooms", ""),
                    ui.input_text_area("num_bathrooms", "Enter Number of Bathrooms", ""),
                    ui.input_action_button("generate_btn", "Generate"),
                    style="background-color: white; border: 2px solid #6180c2; padding: 15px; border-radius: 0 0 5px 5px;"
                ),
            )
        ),
        # Right column with outputs
        ui.column(
            6,
            ui.div(
                ui.h3("Output", style="background-color: #6180c2; color: white; padding: 10px; border-radius: 5px 5px 0 0; margin: 0;"),
                ui.div(
                    ui.h4("Map"),
                    ui.output_ui("map_output"),
                    ui.h4("Predicted Price:"),
                    ui.output_text_verbatim("score_output"),
                    style="background-color: white; border: 2px solid #6180c2; padding: 15px; border-radius: 0 0 5px 5px;"
                ),
            )
        ),
    ),
)

# Server function
def server(input, output, session):
    geolocator = Nominatim(user_agent="shiny_app")
    location_coords = reactive.Value(None)
    prediction_result = reactive.Value("Enter details and click Generate to see the prediction.")

    @reactive.Effect
    @reactive.event(input.generate_btn)
    def generate_output():
        # Generate map location
        address = f"{input.address_input()}, {input.city()}, {input.state()} {input.zip_code()}"
        if address.strip() == ", , ":
            location_coords.set(None)
            prediction_result.set("Invalid address. Please enter a valid address.")
            return

        try:
            location = geolocator.geocode(address)
            if location:
                location_coords.set((location.latitude, location.longitude))
            else:
                location_coords.set(None)
                prediction_result.set("Address not found.")
                return
        except Exception as e:
            location_coords.set(None)
            prediction_result.set(f"Error locating address: {e}")
            return

        # Predict price
        input_features = {feature: None for feature in NUMERICAL_FEATURES + CATEGORICAL_FEATURES}
        try:
            input_features['adTargets/bd'] = float(input.num_bedroom()) if input.num_bedroom().strip() else None
            input_features['bathrooms'] = float(input.num_bathrooms()) if input.num_bathrooms().strip() else None

            # Create DataFrame for prediction
            input_df = pd.DataFrame([input_features])

            # Predict
            prediction = model.predict(input_df)[0]
            prediction_result.set(f"Predicted Price: ${prediction:,.2f}")
        except Exception as e:
            prediction_result.set(f"Error predicting price: {e}")

    @output
    @render.ui
    def map_output():
        coords = location_coords.get()
        if coords is None:
            return ui.p("Enter a valid address and click 'Generate' to see the map and price.")
        else:
            m = folium.Map(location=coords, zoom_start=15)
            folium.Marker(coords, tooltip="Input Address").add_to(m)
            map_html = m._repr_html_()
            return ui.HTML(map_html)

    @output
    @render.text
    def score_output():
        return prediction_result.get()

# Create the Shiny app
app = App(app_ui, server)