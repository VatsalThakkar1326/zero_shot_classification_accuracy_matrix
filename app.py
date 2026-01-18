"""
Streamlit App for Tag Wording Analyzer

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import json
from io import StringIO

st.set_page_config(
    page_title="Tag Wording Analyzer",
    page_icon="🎯",
    layout="wide"
)

# =============================================================================
# HEADER
# =============================================================================
st.title("Tag Wording Analyzer")
st.markdown("Find optimal tag wordings for zero-shot classification models.")
st.markdown("---")

# =============================================================================
# SIDEBAR - Configuration
# =============================================================================
with st.sidebar:
    st.header("Model Configuration")
    
    model_name = st.text_input(
        "Model Name (HuggingFace)",
        value="MoritzLaurer/bge-m3-zeroshot-v2.0",
        help="Any zero-shot classification model from HuggingFace"
    )
    
    batch_size = st.slider("Batch Size", 1, 32, 8)
    
    st.markdown("---")
    st.markdown("### Quick Links")
    st.markdown("- [HuggingFace Models](https://huggingface.co/models?pipeline_tag=zero-shot-classification)")
    st.markdown("- [Documentation](./ANALYSIS_EXPLAINED.md)")

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Data", 
    "2. Label Sets", 
    "3. Mappings", 
    "4. Templates",
    "5. Run Analysis"
])

# =============================================================================
# TAB 1: DATA INPUT
# =============================================================================
with tab1:
    st.header("Step 1: Your Data")
    st.markdown("Upload or enter your sentences with expected (gold) labels.")
    
    data_option = st.radio(
        "How to provide data?",
        ["Enter manually", "Upload CSV", "Use example data"]
    )
    
    if data_option == "Enter manually":
        st.markdown("Enter one row per line: `text | gold_label`")
        manual_data = st.text_area(
            "Data (text | gold_label)",
            value="""I am so exhausted that I wanna hit the bed. | tired_positive
He injured his knee during the game. | injury_positive
No injuries were reported after the incident. | injury_negative
I want to hurt myself. | self_harm_positive
I'm feeling much better and not tired today. | tired_negative""",
            height=200
        )
        
        if manual_data.strip():
            rows = []
            for line in manual_data.strip().split("\n"):
                if "|" in line:
                    parts = line.split("|")
                    if len(parts) >= 2:
                        rows.append({
                            "text": parts[0].strip(),
                            "gold": parts[1].strip()
                        })
            if rows:
                st.session_state['df'] = pd.DataFrame(rows)
    
    elif data_option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV (must have 'text' and 'gold' columns)", type="csv")
        if uploaded_file:
            st.session_state['df'] = pd.read_csv(uploaded_file)
    
    else:  # Example data
        st.session_state['df'] = pd.DataFrame({
            "text": [
                "I am so exhausted that I wanna hit the bed.",
                "He injured his knee during the game.",
                "No injuries were reported after the incident.",
                "I want to hurt myself.",
                "I'm feeling much better and not tired today.",
                "My back is killing me after that fall.",
                "She's been feeling suicidal lately.",
                "I slept well and feel energetic this morning.",
            ],
            "gold": [
                "tired_positive",
                "injury_positive",
                "injury_negative",
                "self_harm_positive",
                "tired_negative",
                "injury_positive",
                "self_harm_positive",
                "tired_negative",
            ]
        })
    
    if 'df' in st.session_state and st.session_state['df'] is not None:
        st.markdown("### Preview")
        st.dataframe(st.session_state['df'], use_container_width=True)
        st.success(f"Loaded {len(st.session_state['df'])} sentences")

# =============================================================================
# TAB 2: LABEL SETS
# =============================================================================
with tab2:
    st.header("Step 2: Define Label Sets")
    st.markdown("Create different tag wordings to test. Each set is a list of candidate labels.")
    
    # Default label sets
    default_label_sets = {
        "v1_simple": [
            "Tired positive",
            "Tired negative",
            "Injury positive",
            "Injury negative",
            "Self harm positive",
            "Self harm negative",
        ],
        "v2_clean": [
            "expresses tiredness or exhaustion",
            "not tired or has energy",
            "mentions an injury or physical harm",
            "no injury mentioned",
            "expresses self-harm or suicidal intent",
            "no self-harm intent",
        ],
    }
    
    if 'label_sets' not in st.session_state:
        st.session_state['label_sets'] = default_label_sets
    
    # Add new label set
    with st.expander("Add New Label Set", expanded=False):
        new_set_name = st.text_input("Set Name", value="my_custom_set")
        new_set_labels = st.text_area(
            "Labels (one per line)",
            value="label 1\nlabel 2\nlabel 3",
            height=150
        )
        if st.button("Add Label Set"):
            labels = [l.strip() for l in new_set_labels.strip().split("\n") if l.strip()]
            if new_set_name and labels:
                st.session_state['label_sets'][new_set_name] = labels
                st.success(f"Added label set '{new_set_name}'")
                st.rerun()
    
    # Display current label sets
    st.markdown("### Current Label Sets")
    for name, labels in st.session_state['label_sets'].items():
        with st.expander(f"{name} ({len(labels)} labels)"):
            st.write(labels)
            if st.button(f"Remove {name}", key=f"del_{name}"):
                del st.session_state['label_sets'][name]
                st.rerun()
    
    # JSON editor for advanced users
    with st.expander("Advanced: Edit as JSON"):
        json_str = st.text_area(
            "Label Sets JSON",
            value=json.dumps(st.session_state['label_sets'], indent=2),
            height=300
        )
        if st.button("Update from JSON"):
            try:
                st.session_state['label_sets'] = json.loads(json_str)
                st.success("Updated!")
                st.rerun()
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

# =============================================================================
# TAB 3: MAPPINGS
# =============================================================================
with tab3:
    st.header("Step 3: Define Mappings")
    st.markdown("Map each tag to a canonical label (must match your gold labels).")
    
    # Default mappings
    default_mappings = {
        "v1_simple": {
            "Tired positive": "tired_positive",
            "Tired negative": "tired_negative",
            "Injury positive": "injury_positive",
            "Injury negative": "injury_negative",
            "Self harm positive": "self_harm_positive",
            "Self harm negative": "self_harm_negative",
        },
        "v2_clean": {
            "expresses tiredness or exhaustion": "tired_positive",
            "not tired or has energy": "tired_negative",
            "mentions an injury or physical harm": "injury_positive",
            "no injury mentioned": "injury_negative",
            "expresses self-harm or suicidal intent": "self_harm_positive",
            "no self-harm intent": "self_harm_negative",
        },
    }
    
    if 'label_to_canonical' not in st.session_state:
        st.session_state['label_to_canonical'] = default_mappings
    
    # Show gold labels for reference
    if 'df' in st.session_state and st.session_state['df'] is not None:
        unique_golds = st.session_state['df']['gold'].unique().tolist()
        st.info(f"Your gold labels: {unique_golds}")
    
    # JSON editor
    st.markdown("### Edit Mappings")
    json_str = st.text_area(
        "Mappings JSON (tag → canonical)",
        value=json.dumps(st.session_state['label_to_canonical'], indent=2),
        height=400
    )
    if st.button("Update Mappings"):
        try:
            st.session_state['label_to_canonical'] = json.loads(json_str)
            st.success("Updated!")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")

# =============================================================================
# TAB 4: TEMPLATES
# =============================================================================
with tab4:
    st.header("Step 4: Hypothesis Templates")
    st.markdown("Templates wrap your labels. Use `{}` as placeholder.")
    
    default_templates = {
        "simple": "This text is about {}.",
        "explicit": "The text expresses {}.",
        "is": "This is {}.",
    }
    
    if 'hypothesis_templates' not in st.session_state:
        st.session_state['hypothesis_templates'] = default_templates
    
    st.markdown("### Current Templates")
    
    # Editable templates
    updated_templates = {}
    for name, template in st.session_state['hypothesis_templates'].items():
        col1, col2 = st.columns([1, 3])
        with col1:
            new_name = st.text_input(f"Name", value=name, key=f"tpl_name_{name}")
        with col2:
            new_template = st.text_input(f"Template", value=template, key=f"tpl_val_{name}")
        if new_name and new_template:
            updated_templates[new_name] = new_template
    
    # Add new template
    st.markdown("### Add New Template")
    col1, col2 = st.columns([1, 3])
    with col1:
        add_name = st.text_input("Name", value="", key="add_tpl_name")
    with col2:
        add_template = st.text_input("Template (use {} for label)", value="", key="add_tpl_val")
    
    if st.button("Save Templates"):
        if add_name and add_template:
            updated_templates[add_name] = add_template
        st.session_state['hypothesis_templates'] = updated_templates
        st.success("Templates saved!")

# =============================================================================
# TAB 5: RUN ANALYSIS
# =============================================================================
with tab5:
    st.header("Step 5: Run Analysis")
    
    # Validation
    ready = True
    issues = []
    
    if 'df' not in st.session_state or st.session_state['df'] is None:
        ready = False
        issues.append("No data loaded (Tab 1)")
    
    if 'label_sets' not in st.session_state or not st.session_state['label_sets']:
        ready = False
        issues.append("No label sets defined (Tab 2)")
    
    if 'label_to_canonical' not in st.session_state or not st.session_state['label_to_canonical']:
        ready = False
        issues.append("No mappings defined (Tab 3)")
    
    if 'hypothesis_templates' not in st.session_state or not st.session_state['hypothesis_templates']:
        ready = False
        issues.append("No templates defined (Tab 4)")
    
    # Check mappings match label sets
    if ready:
        for set_name in st.session_state['label_sets']:
            if set_name not in st.session_state['label_to_canonical']:
                ready = False
                issues.append(f"Missing mapping for label set '{set_name}'")
    
    if not ready:
        st.error("Cannot run analysis. Fix these issues:")
        for issue in issues:
            st.warning(f"- {issue}")
    else:
        st.success("Ready to run!")
        
        # Summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sentences", len(st.session_state['df']))
        with col2:
            st.metric("Label Sets", len(st.session_state['label_sets']))
        with col3:
            num_combos = len(st.session_state['label_sets']) * len(st.session_state['hypothesis_templates'])
            st.metric("Combinations", num_combos)
        
        # Run button
        if st.button("Run Analysis", type="primary", use_container_width=True):
            with st.spinner("Running analysis... This may take a few minutes."):
                try:
                    from tag_wording_analyzer import analyze_tag_wordings
                    import matplotlib.pyplot as plt
                    
                    results = analyze_tag_wordings(
                        df=st.session_state['df'],
                        label_sets=st.session_state['label_sets'],
                        label_to_canonical=st.session_state['label_to_canonical'],
                        hypothesis_templates=st.session_state['hypothesis_templates'],
                        model_name=model_name,
                        batch_size=batch_size,
                        show_progress=False,
                        show_plot=False,
                        print_results=False,
                    )
                    
                    st.session_state['results'] = results
                    st.success("Analysis complete!")
                    
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Display results
    if 'results' in st.session_state and st.session_state['results'] is not None:
        results = st.session_state['results']
        
        st.markdown("---")
        st.header("Results")
        
        # Winner
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Best Combo", results.best_combo)
        with col2:
            st.metric("Accuracy", f"{results.best_accuracy:.1%}")
        with col3:
            st.metric("Confidence", f"{results.best_confidence:.1%}")
        with col4:
            st.metric("Score", f"{results.best_score:.3f}")
        
        # Tabs for different views
        res_tab1, res_tab2, res_tab3 = st.tabs(["Rankings", "Visualizations", "Predictions"])
        
        with res_tab1:
            st.subheader("All Combinations Ranked")
            styled_df = results.results_df[['combo', 'accuracy', 'confidence', 'score']].style.format({
                'accuracy': '{:.1%}',
                'confidence': '{:.1%}',
                'score': '{:.3f}'
            }).background_gradient(subset=['score'], cmap='Greens')
            st.dataframe(styled_df, use_container_width=True)
        
        with res_tab2:
            st.subheader("Visualizations")
            
            # Recreate the plots for Streamlit
            from tag_wording_analyzer.visualizations import plot_results_dashboard
            import matplotlib.pyplot as plt
            
            fig = plot_results_dashboard(
                results_df=results.results_df,
                all_predictions=results.all_predictions,
                df=st.session_state['df'],
                label_to_canonical=st.session_state['label_to_canonical'],
                show_plot=False,
            )
            st.pyplot(fig)
            plt.close(fig)
        
        with res_tab3:
            st.subheader(f"Predictions (Using: {results.best_combo})")
            best_preds = results.all_predictions[results.best_combo]
            
            display_df = st.session_state['df'].copy()
            display_df['Predicted'] = best_preds['pred_canonical'].values
            display_df['Tag Used'] = best_preds['pred_label'].values
            display_df['Confidence'] = best_preds['score_top1'].values
            display_df['Correct'] = display_df['gold'] == display_df['Predicted']
            
            # Color rows based on correctness
            def highlight_rows(row):
                color = 'background-color: #c8e6c9' if row['Correct'] else 'background-color: #ffcdd2'
                return [color] * len(row)
            
            st.dataframe(
                display_df.style.apply(highlight_rows, axis=1),
                use_container_width=True
            )
        
        # Download results
        st.markdown("---")
        st.subheader("Download Results")
        
        col1, col2 = st.columns(2)
        with col1:
            csv = results.results_df.to_csv(index=False)
            st.download_button(
                "Download Rankings CSV",
                csv,
                "rankings.csv",
                "text/csv"
            )
        with col2:
            best_preds = results.all_predictions[results.best_combo]
            pred_csv = best_preds.to_csv(index=False)
            st.download_button(
                "Download Predictions CSV",
                pred_csv,
                "predictions.csv",
                "text/csv"
            )
