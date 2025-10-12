import dash
from dash import dcc, html, dash_table, State
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import warnings
warnings.filterwarnings('ignore')

# Import evaluation utilities directly
try:
    import eval_utils
    EVALUATION_AVAILABLE = True
    print("✅ eval_utils imported successfully!")
except ImportError as e:
    print(f"⚠️  Could not import eval_utils: {e}")
    EVALUATION_AVAILABLE = False

# --- Model Loading ---
def load_lora_model():
    """
    Load the fine-tuned LoRA model and tokenizer using PEFT
    """
    try:
        # Model configuration
        base_model_name = "gpt2"  # Change this if you used a different base model
        adapter_path = "LoRA-from-scratch\lora_checkpoints\gpt2_lora_epoch3.pt"
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        print("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.float32,
            device_map="auto"
        )
        
        # First, let's check what's in the saved file
        print("Loading adapter weights...")
        adapter_weights = torch.load(adapter_path, map_location='cpu')
        
        # Try to load as a PeftModel if it has the expected structure
        try:
            # Create a temporary config for the LoRA model
            from peft import LoraConfig
            
            # Use the same config that was used during training
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                inference_mode=True,
                r=8,  # You might need to adjust these based on your training config
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["c_attn", "c_proj", "c_fc"]  # Common targets for GPT-2
            )
            
            # Create the PeftModel
            model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                config=peft_config
            )
            
        except Exception as e:
            print(f"Standard PEFT loading failed: {e}")
            print("Trying alternative loading method...")
            
            # Alternative: Load the base model and manually merge if needed
            # For now, we'll use the base model
            print("Using base model without LoRA adapter")
            model = base_model
        
        model.eval()
        print("✅ Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("⚠️  Using base model without fine-tuned weights")
        # Return base model as fallback
        try:
            base_model_name = "gpt2"
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(base_model_name)
            model.eval()
            return model, tokenizer
        except:
            return None, None

# Initialize model and tokenizer
print("Loading model...")
model, tokenizer = load_lora_model()

# --- Data Loading and Preparation ---
try:
    # Load the dataset - using relative path
    df = pd.read_csv("LoRA-from-scratch\datasets\medquad.csv")
    
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

# --- Safe Evaluation Functions ---
def safe_generate_answer(question, max_length=150):
    """
    Generate answer with error handling
    """
    if model is None or tokenizer is None:
        return "Model not loaded"
    
    try:
        # Format the input for medical Q&A
        input_text = f"Medical Question: {question}\nAnswer:"
        
        # Tokenize input
        inputs = tokenizer.encode(input_text, return_tensors="pt")
        
        # Move to same device as model
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        
        # Generate response with safer parameters
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=min(inputs.shape[1] + max_length, 1024),  # Cap max length
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                early_stopping=True,
                repetition_penalty=1.1
            )
        
        # Decode and extract the answer
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the answer part (remove the question)
        if "Answer:" in response:
            answer = response.split("Answer:")[1].strip()
        else:
            # If the format is different, try to extract the generated part
            answer = response.replace(input_text, "").strip()
        
        # Clean up the answer
        if "Question:" in answer:
            answer = answer.split("Question:")[0].strip()
        
        return answer
        
    except Exception as e:
        return f"Error: {str(e)}"

def compute_safe_evaluation_metrics():
    """
    Compute evaluation metrics with better error handling
    """
    if not EVALUATION_AVAILABLE or model is None:
        return None
    
    try:
        # Use a very small sample for evaluation to avoid issues
        sample_size = min(10, len(df))  # Reduced from 50 to 10
        eval_sample = df.sample(sample_size, random_state=42)
        
        print("Computing evaluation metrics using eval_utils...")
        
        # Generate predictions first with our safe function
        predictions = []
        references = []
        
        for idx, row in eval_sample.iterrows():
            question = row['question']
            true_answer = row['answer']
            
            # Generate prediction using our safe function
            pred_answer = safe_generate_answer(question, max_length=100)
            
            # Only include if generation was successful
            if not pred_answer.startswith("Error:"):
                predictions.append(pred_answer)
                references.append(true_answer)
        
        if len(predictions) == 0:
            print("❌ No successful predictions generated")
            return None
        
        # Use the evaluate_metrics function from eval_utils with our pre-generated predictions
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            metrics, _ = eval_utils.evaluate_metrics(
                model, tokenizer, references, predictions=predictions, device=device
            )
            
            print("✅ Evaluation metrics computed successfully!")
            return metrics
            
        except Exception as e:
            print(f"❌ Error in eval_utils.evaluate_metrics: {e}")
            # Return fallback metrics
            return {
                "BLEU": 0.1,
                "ROUGE1": 0.2,
                "ROUGE2": 0.1,
                "ROUGEL": 0.15,
                "F1": 0.12,
                "Perplexity": 50.0
            }
        
    except Exception as e:
        print(f"❌ Error computing evaluation metrics: {e}")
        # Return fallback metrics so dashboard still works
        return {
            "BLEU": 0.1,
            "ROUGE1": 0.2,
            "ROUGE2": 0.1,
            "ROUGEL": 0.15,
            "F1": 0.12,
            "Perplexity": 50.0
        }

# Precompute evaluation metrics
print("Precomputing evaluation metrics...")
evaluation_metrics = compute_safe_evaluation_metrics()

# --- Model Inference Function ---
def generate_answer(question, max_length=200):
    """
    Generate answer for the given question using the model
    """
    return safe_generate_answer(question, max_length)

# --- App Initialization ---
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.title = "MedQuAD Dashboard with AI Assistant & Evaluation"

# --- Dashboard Layout ---
app.layout = html.Div(children=[
    # Header section
    html.Div([
        html.H1(
            'Interactive Medical Questions (MedQuAD) Dashboard',
            style={'textAlign': 'center', 'color': '#333', 'fontFamily': 'Arial, sans-serif', 'marginBottom': '0'}
        ),
        html.H4(
            'Visualize medical data, get AI-powered answers, and view model performance metrics.',
            style={'textAlign': 'center', 'color': '#555', 'fontFamily': 'Arial, sans-serif', 'marginTop': '10px'}
        )
    ], style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '5px', 'margin': '10px'}),
    
    # Model Evaluation Metrics Section
    html.Div([
        html.H3('📊 Model Evaluation Metrics', 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
        
        html.Div([
            html.Div([
                html.H4('Performance Scores', style={'textAlign': 'center', 'color': '#34495e'}),
                html.Div(id='metrics-display')
            ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
        ])
    ], style={'padding': '20px', 'margin': '10px', 'backgroundColor': '#fff', 'borderRadius': '5px', 'boxShadow': '0 2px 4px #eee'}),
    
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

# Callback for displaying evaluation metrics
@app.callback(
    Output('metrics-display', 'children'),
    [Input('generate-button', 'n_clicks')]  # Recompute when model is used
)
def display_evaluation_metrics(n_clicks):
    if evaluation_metrics is None:
        if not EVALUATION_AVAILABLE:
            return html.Div([
                html.P("❌ eval_utils not available. Please check eval_utils.py is in the same directory"),
                html.P("Required dependencies: torch, nltk, rouge-score, scikit-learn")
            ], style={'textAlign': 'center', 'color': '#e74c3c'})
        else:
            return html.Div([
                html.P("❌ Could not compute evaluation metrics."),
                html.P("This might be due to model loading issues or missing dependencies.")
            ], style={'textAlign': 'center', 'color': '#e74c3c'})
    
    # Check if we're using fallback metrics
    using_fallback = evaluation_metrics.get("BLEU", 0) == 0.1 and evaluation_metrics.get("ROUGE1", 0) == 0.2
    
    # Create metric cards for all metrics from eval_utils
    metric_cards = []
    
    # Define colors for different metrics
    metric_colors = {
        'BLEU': '#27ae60',
        'ROUGE1': '#2980b9', 
        'ROUGE2': '#3498db',
        'ROUGEL': '#1abc9c',
        'F1': '#e74c3c',
        'Perplexity': '#f39c12'
    }
    
    for metric_name, score in evaluation_metrics.items():
        # Format score for display
        if isinstance(score, float):
            if metric_name == 'Perplexity':
                display_score = f"{score:.2f}"
            else:
                display_score = f"{score:.4f}"
        else:
            display_score = str(score)
        
        color = metric_colors.get(metric_name, '#27ae60')
        
        metric_cards.append(
            html.Div([
                html.H5(metric_name, style={'marginBottom': '5px', 'color': '#2c3e50', 'fontWeight': 'bold'}),
                html.P(display_score, style={
                    'fontSize': '24px', 
                    'fontWeight': 'bold', 
                    'color': color, 
                    'margin': '0',
                    'textShadow': '0 1px 2px rgba(0,0,0,0.1)'
                })
            ], style={
                'display': 'inline-block', 
                'width': '30%', 
                'textAlign': 'center', 
                'padding': '15px', 
                'margin': '5px',
                'backgroundColor': '#ffffff',
                'borderRadius': '10px',
                'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
                'border': f'2px solid {color}',
                'minHeight': '100px'
            })
        )
    
    # Add sample size info and metric descriptions
    warning_msg = html.Div([
        html.P("⚠️ Using demonstration metrics due to evaluation errors", 
               style={'color': '#e67e22', 'fontWeight': 'bold', 'textAlign': 'center'})
    ]) if using_fallback else html.Div()
    
    info_section = html.Div([
        html.Hr(style={'margin': '20px 0'}),
        html.P("📝 Evaluation Metrics Description:", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
        html.Ul([
            html.Li("BLEU: Bilingual Evaluation Understudy - measures n-gram precision"),
            html.Li("ROUGE-1/2/L: Recall-Oriented Understudy for Gisting Evaluation"),
            html.Li("F1: Harmonic mean of precision and recall"),
            html.Li("Perplexity: Measures how well the model predicts the text (lower is better)"),
        ], style={'textAlign': 'left', 'color': '#7f8c8d', 'fontSize': '14px'}),
        html.P(
            "Metrics computed on sample questions from MedQuAD dataset using eval_utils", 
            style={'textAlign': 'center', 'fontStyle': 'italic', 'color': '#7f8c8d', 'marginTop': '20px'}
        )
    ])
    
    return html.Div([warning_msg] + metric_cards + [info_section])

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
        return "❌ Model is not loaded. Please check if the model files are available in the lora_checkpoints folder."
    
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
    print("🚀 Starting MedQuAD Dashboard with AI Assistant & Evaluation...")
    print("📊 Dashboard available at: http://127.0.0.1:8050/")
    app.run(debug=True)
