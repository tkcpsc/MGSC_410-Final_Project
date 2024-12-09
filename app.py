from shiny import App, ui, render, reactive
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Color Palette
PRIMARY_COLOR = "#007BFF"  # Primary color (e.g., blue for headers)
SECONDARY_COLOR = "#D3D3D3"  # Secondary color (e.g., for pie chart empty parts)
WHITE_COLOR = "#ffffff"  # Secondary color (e.g., for pie chart empty parts)
BACKGROUND_COLOR = "#2e2e2e"  # Background color for cards and figure
ACCENT_COLOR = "#FFD700"  # Accent color (e.g., for hover or additional highlights)
BODY_BACKGROUND = "#212121"  # Background color for the entire page

# User Interface
app_ui = ui.page_fluid(
    ui.tags.style(f"""
        body {{
            background-color: {BODY_BACKGROUND};  /* Set the body background color */
            color: white;  /* Default text color for dark background */
            font-family: Arial, sans-serif;
        }}
        .dashboard {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
            padding: 20px;
        }}
        .card {{
            background-color: {BACKGROUND_COLOR};
            border-radius: 10px;
            width: 300px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
            color: white;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .card-header {{
            background-color: {PRIMARY_COLOR};
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            font-size: 1.2em;
            font-weight: bold;
        }}
        .card-content {{
            flex: 1;
            padding: 10px;
            font-size: 0.9em;
            max-height: 300px;  /* Set maximum height for card content */
            overflow-y: auto;  /* Enable vertical scrolling if content overflows */
        }}
    """),
    ui.div(
        ui.div(
            ui.div(
                ui.div("Input Field", class_="card-header"),
                ui.div(ui.input_text("input_text", "S&P 500 Stock Ticker:", value=""), class_="card-content"),
                ui.div(ui.input_action_button("generate_btn", "Generate"), style="text-align: center; margin-top: 10px;"),
                class_="card",
            ),
            ui.div(
                ui.div("Forecast Prediction", class_="card-header"),
                ui.div(ui.output_ui("forecast_output"), class_="card-content"),
                class_="card",
            ),
            ui.div(
                ui.div("Current Stock Data", class_="card-header"),
                ui.div(ui.output_ui("stock_data_output"), class_="card-content"),
                class_="card",
            ),
            ui.div(
                ui.div("LLM Interpretation", class_="card-header"),
                ui.div(ui.output_ui("llm_interpretation_output"), class_="card-content"),
                class_="card",
            ),
            ui.div(
                ui.div("Article 1", class_="card-header"),
                ui.div(ui.output_ui("article_1_output"), class_="card-content"),
                class_="card",
            ),
            ui.div(
                ui.div("Article 2", class_="card-header"),
                ui.div(ui.output_ui("article_2_output"), class_="card-content"),
                class_="card",
            ),
            ui.div(
                ui.div("Article 3", class_="card-header"),
                ui.div(ui.output_ui("article_3_output"), class_="card-content"),
                class_="card",
            ),
            ui.div(
                ui.div("Article 4", class_="card-header"),
                ui.div(ui.output_ui("article_4_output"), class_="card-content"),
                class_="card",
            ),
            ui.div(
                ui.div("Article 5", class_="card-header"),
                ui.div(ui.output_ui("article_5_output"), class_="card-content"),
                class_="card",
            ),
            ui.div(
                ui.div("Article 6", class_="card-header"),
                ui.div(ui.output_ui("article_6_output"), class_="card-content"),
                class_="card",
            ),
            ui.div(
                ui.div("Article 7", class_="card-header"),
                ui.div(ui.output_ui("article_7_output"), class_="card-content"),
                class_="card",
            ),
            ui.div(
                ui.div("Article 8", class_="card-header"),
                ui.div(ui.output_ui("article_8_output"), class_="card-content"),
                class_="card",
            ),
            ui.div(
                ui.div("Article 9", class_="card-header"),
                ui.div(ui.output_ui("article_9_output"), class_="card-content"),
                class_="card",
            ),
            ui.div(
                ui.div("Article 10", class_="card-header"),
                ui.div(ui.output_ui("article_10_output"), class_="card-content"),
                class_="card",
            ),
            ui.div(
                ui.div("Article 11", class_="card-header"),
                ui.div(ui.output_ui("article_11_output"), class_="card-content"),
                class_="card",
            ),
            ui.div(
                ui.div("Article 12", class_="card-header"),
                ui.div(ui.output_ui("article_12_output"), class_="card-content"),
                class_="card",
            ),
            ui.div(
                ui.div("Article 13", class_="card-header"),
                ui.div(ui.output_ui("article_13_output"), class_="card-content"),
                class_="card",
            ),
            ui.div(
                ui.div("Article 14", class_="card-header"),
                ui.div(ui.output_ui("article_14_output"), class_="card-content"),
                class_="card",
            ),
            class_="dashboard",
        )
    )
)



from input_ticker_obj import StockData  # Import the StockData class
from shiny import App, ui, render, reactive
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def server(input, output, session):
    # Reactive value to store the StockData object
    stock_data = reactive.Value(None)

    @reactive.Effect
    @reactive.event(input.generate_btn)
    def handle_generate_click():
        logging.info("Hi")  # Log "Hi" when the button is clicked

        # Fetch the ticker from the user input
        ticker = input.input_text()
        print(f"Input Ticker: {ticker}")  # Debugging

        if not ticker:
            print("Ticker input is empty.")
            stock_data.set(None)
            return

        try:
            # Fetch StockData (replace with actual logic)
            from input_ticker_obj import StockData
            data = StockData(ticker)
            stock_data.set(data)
            print(f"Data fetched successfully for ticker: {ticker}")
        except Exception as e:
            print(f"Error fetching data for ticker {ticker}: {e}")
            stock_data.set(None)

    # Render stock data output (numerical and general data)
    @output
    @render.ui
    def stock_data_output():
        data = stock_data.get()
        if data is not None and hasattr(data, "df") and not data.df.empty:
            stock_info = "<br>".join(
                f"{col}: {data.df[col].iloc[0]}" for col in data.df.columns if not col.startswith("Article")
            )
            return ui.div(ui.HTML(stock_info), class_="card-content")
        else:
            return ui.div("No data available.", class_="card-content")
    # Render dynamic article outputs (Article 1 to Article 14)
    def render_article_output(article_num):
        @output(id=f"article_{article_num}_output")
        @render.ui
        def dynamic_article_output():
            data = stock_data.get()
            if data is not None and hasattr(data, "get_article_with_attributes"):
                # Use `get_article_with_attributes` to fetch attributes and article text
                article_info = data.get_article_with_attributes(article_num)
                if isinstance(article_info, str):
                    # Split the text while preserving empty lines
                    lines = article_info.splitlines(keepends=True)
                    return ui.div(
                        [
                            ui.div(line.strip(), class_="attribute-line") if line.strip() else ui.div(" ", class_="empty-line")
                            for line in lines
                        ],
                        class_="card-content"
                    )
                else:
                    return ui.div(f"No article available for Article {article_num}.", class_="card-content")
            else:
                return ui.div(f"No article available for Article {article_num}.", class_="card-content")

        return dynamic_article_output 
       
    # Dynamically create renderers for Article 1 to Article 5
    for i in range(1, 15):
        render_article_output(i)

    # Placeholder outputs for unused cards
    @output
    @render.ui
    def forecast_output():
        return ui.div("Forecast feature is under development.", class_="card-content")

    @output
    @render.ui
    def llm_interpretation_output():
        return ui.div("LLM interpretation is under development.", class_="card-content")        
        
        
# Create the Shiny app
app = App(app_ui, server)