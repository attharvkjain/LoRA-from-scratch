import dash
from dash import dcc, html, dash_table, State
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import warnings
warnings.filterwarnings('ignore')

# --- Model Loading ---
def load_lora_model():
    """
    Load the fine-tuned LoRA model and tokenizer
    """
    try:
        # Model configuration - adjust based on your base model
        model_name = "microsoft/DialoGPT-medium"  # Change this to your base model
        peft_model_id = "checkpoints/lora"  # Path to your LoRA checkpoint
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, peft_model_id)
        model.eval()
        
        print("✅ Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

# Initialize model and tokenizer
print("Loading model...")
model, tokenizer = load_lora_model()

# --- Data Loading and Preparation ---
try:
    # Load the dataset
    df = pd.read_csv("datasets/medquad.csv")
    
    # Drop rows with missing values in key columns to ensure clean visualizations
    df.dropna(subset=['source', 'focus_area'], inplace=True)
    
    # Get unique values for dropdown filters
    source_options = [{'label': 'All Sources', 'value': 'all'}] + \
                     [{'label': i, 'value': i} for i in sorted(df['source'].unique())]
                     
    focus_area_options = [{'label': 'All Focus Areas', 'value': 'all'}] + \
                        [{'label': i, 'value': i} for i in sorted(df['focus_area'].unique())]

except FileNotFoundError:
    print("Error: 'medquad.csv' not found. Please make sure the file is in the correct directory.")
    # Create an empty dataframe to prevent the app from crashing
    df = pd.DataFrame({
        'question': [], 'answer': [], 'source': [], 'focus_area': []
    })
    source_options = [{'label': 'No Data', 'value': 'all'}]
    focus_area_options = [{'label': 'No Data', 'value': 'all'}]

# --- Model Inference Function ---
def generate_answer(question, max_length=200):
    """
    Generate answer for the given question using the fine-tuned model
    """
    if model is None or tokenizer is None:
        return "❌ Model not loaded. Please check the model files."
    
    try:
        # Format the input for medical Q&A
        input_text = f"Question: {question}\nAnswer:"
        
        # Tokenize input
        inputs = tokenizer.encode(input_text, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=torch.ones_like(inputs)
            )
        
        # Decode and extract the answer
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the answer part (remove the question)
        if "Answer:" in response:
            answer = response.split("Answer:")[1].strip()
        else:
            answer = response.replace(input_text, "").strip()
        
        return answer
        
    except Exception as e:
        return f"❌ Error generating answer: {str(e)}"

# --- App Initialization ---
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.title = "MedQuAD Dashboard with AI Assistant"

# --- Dashboard Layout ---
app.layout = html.Div(children=[
    # Header section
    html.Div([
        html.H1(
            'Interactive Medical Questions (MedQuAD) Dashboard',
            style={'textAlign': 'center', 'color': '#333', 'fontFamily': 'Arial, sans-serif', 'marginBottom': '0'}
        ),
        html.H4(
            'Visualize medical data and get AI-powered answers to medical questions.',
            style={'textAlign': 'center', 'color': '#555', 'fontFamily': 'Arial, sans-serif', 'marginTop': '10px'}
        )
    ], style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px', 'margin': '10px'}),
    
    # AI Assistant Section
    html.Div([
        html.H3('🤖 Medical AI Assistant', 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
        
        html.Div([
            html.Label('Ask a medical question:', style={'fontWeight': 'bold', 'marginBottom': '10px'}),
            dcc.Textarea(
                id='question-input',
                placeholder='Type your medical question here...\nExample: What are the symptoms of diabetes?',
                style={'width': '100%', 'height': 100, 'padding': '10px', 'fontSize': '16px', 'borderRadius': '5px', 'border': '1px solid #ddd'},
            ),
            html.Button(
                'Get Answer', 
                id='generate-button', 
                n_clicks=0,
                style={'marginTop': '10px', 'padding': '10px 20px', 'backgroundColor': '#0074D9', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}
            ),
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            html.H4('AI Response:', style={'color': '#2c3e50', 'marginBottom': '10px'}),
            html.Div(
                id='answer-output',
                style={'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px', 'border': '1px solid #e9ecef', 'minHeight': '80px', 'whiteSpace': 'pre-wrap'}
            )
        ])
    ], style={'padding': '20px', 'margin': '10px', 'backgroundColor': '#fff', 'borderRadius': '5px', 'boxShadow': '0 2px 4px #eee'}),
   
    # Control panel for filters
    html.Div([
        html.Div([
            html.Label('Filter by Source:', style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='source-filter-dropdown',
                options=source_options,
                value='all', # Default value
                clearable=False
            )
        ], className='six columns'),
        
        html.Div([
            html.Label('Filter by Focus Area:', style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='focus-area-filter-dropdown',
                options=focus_area_options,
                value='all', # Default value
                clearable=False
            )
        ], className='six columns'),
    ], className='row', style={'padding': '10px', 'margin': '10px', 'backgroundColor': '#fff', 'borderRadius': '5px', 'boxShadow': '0 2px 4px #eee'}),

    # Graphs section
    html.Div([
        # Left graph: Questions per Source
        html.Div([
            dcc.Graph(id='source-bar-chart')
        ], className='six columns'),
        
        # Right graph: Questions per Focus Area
        html.Div([
            dcc.Graph(id='focus-area-pie-chart')
        ], className='six columns'),
    ], className='row', style={'padding': '10px'}),
    
    # Data table section
    html.Div([
        html.H3("Filtered Data", style={'textAlign': 'center', 'fontFamily': 'Arial, sans-serif', 'marginTop': '20px'}),
        dash_table.DataTable(
            id='data-table',
            columns=[
                {'name': 'Question', 'id': 'question'},
                {'name': 'Answer', 'id': 'answer'},
                {'name': 'Source', 'id': 'source'},
                {'name': 'Focus Area', 'id': 'focus_area'}
            ],
            page_size=10,
            style_cell={'textAlign': 'left', 'fontFamily': 'Arial, sans-serif', 'padding': '10px'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data={
                'whiteSpace': 'normal',
                'height': 'auto',
            },
            filter_action="native",
            sort_action="native",
        )
    ], style={'padding': '20px', 'margin': '10px', 'backgroundColor': '#fff', 'borderRadius': '5px', 'boxShadow': '0 2px 4px #eee'})

], style={'maxWidth': '1200px', 'margin': 'auto'})

# --- Callbacks for Interactivity ---

# Callback for AI Assistant
@app.callback(
    Output('answer-output', 'children'),
    [Input('generate-button', 'n_clicks')],
    [State('question-input', 'value')]
)
def generate_medical_answer(n_clicks, question):
    if n_clicks == 0 or not question:
        return "Ask a medical question above to get an AI-generated answer..."
    
    if model is None:
        return "❌ Model is not loaded. Please check if the model files are available in the checkpoints folder."
    
    # Show loading message
    answer = "⏳ Generating answer... Please wait."
    
    # Generate the actual answer
    answer = generate_answer(question)
    
    return answer

# Callback for dashboard filters and visualizations
@app.callback(
    [Output('source-bar-chart', 'figure'),
     Output('focus-area-pie-chart', 'figure'),
     Output('data-table', 'data')],
    [Input('source-filter-dropdown', 'value'),
     Input('focus-area-filter-dropdown', 'value')]
)
def update_dashboard(selected_source, selected_focus_area):
    # Start with the full dataframe
    filtered_df = df.copy()

    # Apply filters based on dropdown selections
    if selected_source != 'all':
        filtered_df = filtered_df[filtered_df['source'] == selected_source]
    
    if selected_focus_area != 'all':
        filtered_df = filtered_df[filtered_df['focus_area'] == selected_focus_area]

    # --- Create Bar Chart for Sources ---
    source_counts = filtered_df['source'].value_counts().reset_index()
    source_counts.columns = ['Source', 'Count']
    source_bar_fig = px.bar(
        source_counts.head(15), # Show top 15 sources
        x='Source',
        y='Count',
        title='Number of Questions per Source',
        labels={'Count': 'Number of Questions'},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    source_bar_fig.update_layout(transition_duration=500, title_x=0.5)

    # --- Create Pie Chart for Focus Areas ---
    focus_area_counts = filtered_df['focus_area'].value_counts().reset_index()
    focus_area_counts.columns = ['Focus Area', 'Count']
    focus_area_pie_fig = px.pie(
        focus_area_counts,
        names='Focus Area',
        values='Count',
        title='Distribution of Focus Areas',
        hole=0.3 # Make it a donut chart
    )
    focus_area_pie_fig.update_traces(textposition='inside', textinfo='percent+label')
    focus_area_pie_fig.update_layout(transition_duration=500, title_x=0.5, showlegend=False)

    # --- Prepare data for the table ---
    table_data = filtered_df[['question', 'answer', 'source', 'focus_area']].to_dict('records')
    
    return source_bar_fig, focus_area_pie_fig, table_data

# --- Run the App ---
if __name__ == '__main__':
    # The server will run on http://127.0.0.1:8050/ by default
    print("🚀 Starting MedQuAD Dashboard with AI Assistant...")
    print("📊 Dashboard available at: http://127.0.0.1:8050/")
    app.run(debug=True)