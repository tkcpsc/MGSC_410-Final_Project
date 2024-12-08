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
    # Single Card
    ui.card(
        # Card Title
        ui.card_header(
            ui.h2("AI-Powered News Analysis for Stock Market Forecasting", style=f"color: white; margin: 0; text-align: center;"),
            style=f"background-color: {PRIMARY_COLOR}; padding: 10px;",  # Dynamic primary color
        ),
        # Card Content
        ui.div(
            ui.output_ui("circle_output"),  # Placeholder for the chart
            ui.div(
                ui.output_text("percentage_label"),
                style="text-align: center; margin-top: 0px;",  # Reduced margin-top for label
            ),
            # Input text box and generate button
            ui.div(
                ui.div(
                    ui.input_text("input_text", "S&P 500 Stock Ticker:", value=""),
                    style=f"color: {WHITE_COLOR}; margin-top: 10px; width: 80%; text-align: center;",
                ),
                ui.div(
                    ui.input_action_button("generate_btn", "Generate"),
                    style=f"background-color: {PRIMARY_COLOR}; color: white; margin-top: 10px; width: 50%; text-align: center; border-",
                ),
                style="display: flex; flex-direction: column; align-items: center; margin-top: 10px;",  # Center inputs
            ),
            style="display: flex; flex-direction: column; align-items: center; padding: 10px;",  # Reduced padding
        ),
        style=(
            f"background-color: {BACKGROUND_COLOR}; "
            "border-radius: 10px; "
            "padding: 0px; "  # Reduced padding
            "max-width: 700px; "  # Set max width
            "margin: 0 auto;"  # Center the card
        ),
    ),
    # Add CSS for responsiveness and color palette
    ui.tags.style(f"""
        body {{
            background-color: {BODY_BACKGROUND};  /* Set the body background color */
            color: white;  /* Default text color for dark background */
        }}
        #circle_output img {{
            width: 50vw;  /* 50% of the viewport width */
            height: auto; /* Maintain aspect ratio */
            max-width: 500px; /* Limit the maximum size */
        }}
        #percentage_label {{
            font-size: 1.5vw; /* Relative font size to viewport width */
            max-font-size: 20px; /* Limit the maximum font size */
            color: {PRIMARY_COLOR}; /* Primary color for the label */
        }}
    """),
)

# Server logic
def server(input, output, session):
    # Reactive value to hold the percentage
    percentage = reactive.Value(50)  # Initialized to 50% for this example

    # Render the pie chart
    @output
    @render.ui
    def circle_output():
        # Dynamic size based on percentage of viewport width
        figure_width = 6  # Use a fixed size for rendering the image
        fig, ax = plt.subplots(figsize=(figure_width, figure_width), subplot_kw={'aspect': 'equal'})
        
        # Set figure background color
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        ax.set_facecolor(BACKGROUND_COLOR)  # Set the axes background color to match
        
        wedges, _ = ax.pie(
            [percentage.get(), 100 - percentage.get()],  # Split circle based on percentage
            startangle=90,
            colors=[PRIMARY_COLOR, SECONDARY_COLOR],  # Use primary and secondary colors
            wedgeprops={'edgecolor': 'white'},  # Add border between segments
        )
        
        # Adjust layout to prevent title cutoff
        fig.tight_layout()

        # Encode the chart as a base64 PNG
        buf = io.BytesIO()
        FigureCanvas(fig).print_png(buf)
        plt.close(fig)
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Return as HTML with embedded image
        img_src = f"data:image/png;base64,{img_base64}"
        return ui.HTML(f'<img src="{img_src}" id="circle_output">')

    # Render the percentage label below the chart
    @output
    @render.text
    def percentage_label():
        return f"Selected Percentage: {percentage.get()}%"

# Create the Shiny app
app = App(app_ui, server)
