from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
import pandas as pd
import numpy as np
import os
from portfolio_utils import calculate_performance_metrics # To calculate SPY metrics

SLIDES_DIR = 'slides'
TEMPLATE_PATH = os.path.join(SLIDES_DIR, 'Module2_template.pptx')
OUTPUT_DIR = 'outputs'
CUM_RET_PNG = os.path.join(OUTPUT_DIR, 'cum_returns.png')
BEST_ALG_PATH = os.path.join(OUTPUT_DIR, 'best_alg.txt')
METRICS_PATH = os.path.join(OUTPUT_DIR, 'metrics.csv')
PERF_SUMMARY_PATH = os.path.join(OUTPUT_DIR, 'perf_summary.csv')
SPY_DATA_PATH = 'Data/SPY returns.xlsx'

def get_best_model_name():
    try:
        with open(BEST_ALG_PATH, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: {BEST_ALG_PATH} not found. Run select_best_model.py first.")
        return None

def load_model_metrics(best_model_name):
    try:
        metrics_df = pd.read_csv(METRICS_PATH)
        model_r2 = metrics_df[metrics_df['model'] == best_model_name]['oos_r2'].iloc[0]
        return model_r2
    except (FileNotFoundError, IndexError, KeyError) as e:
        print(f"Error loading R2 for {best_model_name} from {METRICS_PATH}: {e}")
        return np.nan

def load_performance_summary(best_model_name):
    try:
        summary_df = pd.read_csv(PERF_SUMMARY_PATH)
        model_perf = summary_df[summary_df['model'] == best_model_name].iloc[0].copy() # Use .copy() to avoid SettingWithCopyWarning
        # These will be calculated from the model's actual returns later and added
        if 'avg_return' not in model_perf:
            model_perf['avg_return'] = np.nan 
        if 'volatility' not in model_perf: 
            model_perf['volatility'] = np.nan
        return model_perf
    except (FileNotFoundError, IndexError, KeyError) as e:
        print(f"Error loading performance summary for {best_model_name} from {PERF_SUMMARY_PATH}: {e}")
        return pd.Series(dtype='object') # Return empty series on error to allow .get later

def get_spy_performance_for_period(model_returns_df):
    try:
        spy_returns_raw = pd.read_excel(SPY_DATA_PATH, index_col=0)
        spy_returns_raw.index = pd.to_datetime(spy_returns_raw.index)
        spy_returns_raw.columns = ['spy_ret'] 

        if model_returns_df is None or model_returns_df.empty:
            print("Model returns are empty, cannot align SPY data. Calculating SPY metrics over its entire history.")
            spy_overall_perf = calculate_performance_metrics(spy_returns_raw, spy_returns_raw, 'SPY_overall')
            return pd.Series({
                'alpha': '—', 
                'sharpe': f"{spy_overall_perf.get('sharpe', np.nan).iloc[0]:.2f}" if pd.notna(spy_overall_perf.get('sharpe', np.nan).iloc[0]) else 'N/A',
                'avg_return': f"{spy_returns_raw['spy_ret'].mean():.4f}",
                'volatility': f"{spy_returns_raw['spy_ret'].std():.4f}",
                'max_drawdown': f"{spy_overall_perf.get('max_drawdown', np.nan).iloc[0]:.4f}" if pd.notna(spy_overall_perf.get('max_drawdown', np.nan).iloc[0]) else 'N/A',
                'max_loss': f"{spy_overall_perf.get('max_loss', np.nan).iloc[0]:.4f}" if pd.notna(spy_overall_perf.get('max_loss', np.nan).iloc[0]) else 'N/A'
            })

        model_start_date = model_returns_df.index.min()
        model_end_date = model_returns_df.index.max()
        spy_aligned = spy_returns_raw[(spy_returns_raw.index >= model_start_date) & (spy_returns_raw.index <= model_end_date)]
        
        if spy_aligned.empty:
            print("SPY returns have no overlap with model OOS period.")
            return pd.Series(dtype='object')
        
        spy_perf_metrics = calculate_performance_metrics(spy_aligned, spy_aligned, 'SPY') 

        return pd.Series({
            'alpha': '—', 
            'sharpe': f"{spy_perf_metrics.get('sharpe', np.nan).iloc[0]:.2f}" if pd.notna(spy_perf_metrics.get('sharpe', np.nan).iloc[0]) else 'N/A',
            'avg_return': f"{spy_aligned['spy_ret'].mean():.4f}",
            'volatility': f"{spy_aligned['spy_ret'].std():.4f}",
            'max_drawdown': f"{spy_perf_metrics.get('max_drawdown', np.nan).iloc[0]:.4f}" if pd.notna(spy_perf_metrics.get('max_drawdown', np.nan).iloc[0]) else 'N/A',
            'max_loss': f"{spy_perf_metrics.get('max_loss', np.nan).iloc[0]:.4f}" if pd.notna(spy_perf_metrics.get('max_loss', np.nan).iloc[0]) else 'N/A'
        })
    except Exception as e:
        print(f"Error calculating SPY performance: {e}")
        return pd.Series(dtype='object')

def update_presentation():
    best_model_name = get_best_model_name()
    if not best_model_name:
        print("Best model name not found. Exiting presentation update.")
        return

    oos_r2 = load_model_metrics(best_model_name)
    model_perf_summary = load_performance_summary(best_model_name)

    model_port_ret_path = os.path.join(OUTPUT_DIR, f'{best_model_name}_overall_port_ret.csv')
    model_returns_df = None
    avg_model_monthly_return = np.nan
    std_model_monthly_vol = np.nan

    try:
        model_returns_df = pd.read_csv(model_port_ret_path, index_col=0, parse_dates=True)
        model_returns_col = model_returns_df.columns[0]
        avg_model_monthly_return = model_returns_df[model_returns_col].mean()
        std_model_monthly_vol = model_returns_df[model_returns_col].std()
        if model_perf_summary is not None and not model_perf_summary.empty: # Check if Series is not empty
            model_perf_summary['avg_return'] = avg_model_monthly_return
            model_perf_summary['volatility'] = std_model_monthly_vol
    except FileNotFoundError:
        print(f"Model returns file not found: {model_port_ret_path}")
        # model_perf_summary might be None or empty here, .get will handle it.
        
    spy_perf_for_table = get_spy_performance_for_period(model_returns_df)

    METHODOLOGY_TEXT = (
        f"• Evaluated Models: Lasso, Ridge, ElasticNet, Decision Tree, Neural Network (NN2), IPCA, and Autoencoder.\\n"
        f"• Best Performing Model: {best_model_name.upper()} selected for predicting next-month stock excess returns.\\n"
        f"• Training Window: Expanding, initial 10 years (2005-2014).\\n"
        f"• Validation Window: Expanding, initial 2 years (2015-2016) for hyperparameter tuning/model selection per window.\\n"
        f"• Out-of-Sample (OOS) Testing: Expanding window (2017-2023), yearly model re-estimation/re-training.\\n"
        f"• Portfolio Construction: Equal-weighted top 50 stocks (based on predictions), rebalanced monthly.\\n"
        f"• Features: 145 lagged stock characteristics (time t) predicting returns (time t+1).\\n"
        f"• Benchmark: S&P 500 (SPY)."
    )
    
    oos_r2_title_text = f"Out-of-Sample $R^2$ ({best_model_name.upper()}): {oos_r2:.4f}"
    
    oos_r2_interpretation_text = "Interpretation:\\n• R² measures the proportion of return variance explained by the model.\\n"
    if pd.notna(oos_r2):
        if oos_r2 < 0:
            oos_r2_interpretation_text += (
                f"• Overall Negative R² ({oos_r2:.4f}): Indicates the model, on average, underperformed a simple mean prediction across the OOS period.\\n"
                f"• Influencing Factors: Performance was significantly impacted by specific OOS years (e.g., 2017 for {best_model_name.upper()}), highlighting challenges in consistent prediction."
            )
        else:
            oos_r2_interpretation_text += (
                f"• R² of {oos_r2:.4f}: Suggests the model explains {oos_r2*100:.2f}% of the variance in returns. A positive R² is preferred."
            )
    else:
        oos_r2_interpretation_text += "• R² value not available."

    summary_text = "Summary data not available."
    if model_perf_summary is not None and not model_perf_summary.empty and pd.notna(oos_r2):
        summary_text = (
            f"The {best_model_name.upper()} strategy, despite a challenging overall OOS R² of {oos_r2:.4f} (impacted by specific years), demonstrated notable portfolio performance:\\n"
            f"  • Monthly Alpha: {model_perf_summary.get('alpha', np.nan):.4f}\\n"
            f"  • Annualized Sharpe Ratio: {model_perf_summary.get('sharpe', np.nan):.2f}\\n"
            f"  • Avg. Monthly Return: {model_perf_summary.get('avg_return', np.nan)*100:.2f}%\\n"
            f"  • Monthly Volatility: {model_perf_summary.get('volatility', np.nan)*100:.2f}%\\n"
            f"  • Max Drawdown: {model_perf_summary.get('max_drawdown', np.nan):.2%}\\n"
            f"Potential Improvements:\\n"
            f"• Enhance predictive stability (e.g., robust hyperparameter tuning for {best_model_name.upper()}, address outlier year performance).\\n"
            f"• Explore advanced feature engineering and selection techniques.\\n"
            f"• Investigate alternative portfolio construction or risk management methodologies."
        )

    prs = Presentation(TEMPLATE_PATH)
    
    # Slide 1: Methodology
    slide = prs.slides[0]
    if len(slide.placeholders) > 1:
        tf = slide.placeholders[1].text_frame
        tf.clear()
        p = tf.add_paragraph()
        p.text = METHODOLOGY_TEXT.replace("\\n", "\n") # Ensure newlines are rendered
        p.font.size = Pt(16) 
    
    # Slide 2: OOS R2
    slide = prs.slides[1]
    tf = slide.shapes.placeholders[1].text_frame
    tf.clear()
    p = tf.add_paragraph()
    p.text = oos_r2_title_text
    p.font.size = Pt(30) # Adjusted size
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER
    
    tf.add_paragraph().text = "" # Add a blank line for spacing
    
    p = tf.add_paragraph()
    p.text = oos_r2_interpretation_text.replace("\\n", "\n")
    p.font.size = Pt(18)
    
    # Slide 3: Performance Table
    slide = prs.slides[2]
    for shape in slide.shapes:
        if shape.has_table:
            sp = shape
            slide.shapes._spTree.remove(sp._element)
            
    left = Inches(0.5); top = Inches(1.5); width = Inches(9.0); height = Inches(0.5) # Initial height, will adjust
    
    # Check if model_perf_summary and spy_perf_for_table are valid Series and have data
    model_data_valid = isinstance(model_perf_summary, pd.Series) and not model_perf_summary.empty
    spy_data_valid = isinstance(spy_perf_for_table, pd.Series) and not spy_perf_for_table.empty

    if model_data_valid and spy_data_valid:
        perf_data = [
            ["Metric", f"{best_model_name.upper()} Strategy", "SPY"],
            ["Alpha (monthly)", f"{model_perf_summary.get('alpha', np.nan):.4f}", spy_perf_for_table.get('alpha', 'N/A')],
            ["Sharpe Ratio (ann.)", f"{model_perf_summary.get('sharpe', np.nan):.2f}", spy_perf_for_table.get('sharpe', 'N/A')],
            ["Avg Return (monthly)", f"{model_perf_summary.get('avg_return', np.nan)*100:.2f}%", f"{float(spy_perf_for_table.get('avg_return', np.nan))*100:.2f}%" if pd.notna(spy_perf_for_table.get('avg_return', np.nan)) else 'N/A'],
            ["Volatility (monthly)", f"{model_perf_summary.get('volatility', np.nan)*100:.2f}%", f"{float(spy_perf_for_table.get('volatility', np.nan))*100:.2f}%" if pd.notna(spy_perf_for_table.get('volatility', np.nan)) else 'N/A'],
            ["Max Drawdown", f"{model_perf_summary.get('max_drawdown', np.nan):.2%}", f"{float(spy_perf_for_table.get('max_drawdown', np.nan)):.2%}" if pd.notna(spy_perf_for_table.get('max_drawdown', np.nan)) else 'N/A'],
            ["Max 1-mo Loss", f"{model_perf_summary.get('max_loss', np.nan):.2%}", f"{float(spy_perf_for_table.get('max_loss', np.nan)):.2%}" if pd.notna(spy_perf_for_table.get('max_loss', np.nan)) else 'N/A']
        ]
        rows, cols = len(perf_data), len(perf_data[0])
        table_shape = slide.shapes.add_table(rows, cols, left, top, width, Inches(rows * 0.45)) # Adjusted height per row
        table = table_shape.table
        for r, row_data in enumerate(perf_data):
            for c, cell_data in enumerate(row_data):
                cell = table.cell(r,c)
                cell.text = str(cell_data)
                tc = cell.text_frame
                p = tc.paragraphs[0]
                p.font.size = Pt(14)
                p.alignment = PP_ALIGN.CENTER
                if r == 0: p.font.bold = True
                # Adjust column widths (example: make first column wider)
                if c == 0 : table.columns[c].width = Inches(2.5)
                else: table.columns[c].width = Inches(2.1)


    else:
        tf = slide.shapes.placeholders[1].text_frame; tf.clear(); 
        p = tf.add_paragraph()
        p.text = "Performance data for model or SPY is not available."
        p.alignment = PP_ALIGN.CENTER
        p.font.size = Pt(18)

    # Slide 4: Cumulative Returns Plot
    slide = prs.slides[3]
    # Clear existing title placeholder text if it's the default one
    title_shape = slide.shapes.title
    if title_shape and title_shape.has_text_frame and title_shape.text_frame.text.startswith("Click to add title"):
         title_shape.text_frame.clear() # Clear default text only
         title_shape.text = f"Cumulative OOS Returns: {best_model_name.upper()} vs SPY"


    if os.path.exists(CUM_RET_PNG):
        pic_left = Inches(0.5)
        pic_top = Inches(1.2) # Adjusted top to give space for title
        pic_width = Inches(9.0) 
        # Optional: Maintain aspect ratio
        # from pptx.util import Emu
        # img = Image.open(CUM_RET_PNG)
        # aspect_ratio = img.height / img.width
        # pic_height = Emu(pic_width * aspect_ratio)

        # Clear existing pictures first
        for shape_idx in reversed(range(len(slide.shapes))): # Iterate backwards for safe removal
            shape = slide.shapes[shape_idx]
            if shape.shape_type == 13: # msoPicture
                 sp = shape._element
                 slide.shapes._spTree.remove(sp)
        slide.shapes.add_picture(CUM_RET_PNG, pic_left, pic_top, width=pic_width) # height can be specified too
    else:
        tf = slide.placeholders[1].text_frame if len(slide.placeholders) > 1 else slide.shapes.add_textbox(Inches(1), Inches(1), Inches(8), Inches(1)).text_frame
        tf.clear(); 
        p = tf.add_paragraph()
        p.text = "Cumulative returns plot not found."
        p.alignment = PP_ALIGN.CENTER
        p.font.size = Pt(18)
    
    # Slide 5: Summary
    slide = prs.slides[4]
    if len(slide.placeholders) > 1:
        tf = slide.placeholders[1].text_frame
        tf.clear()
        p = tf.add_paragraph()
        p.text = summary_text.replace("\\n", "\n")
        p.font.size = Pt(16) 
    
    output_ppt_path = os.path.join(OUTPUT_DIR, f'Module3_Report_{best_model_name}.pptx')
    prs.save(output_ppt_path)
    print(f"Presentation updated and saved to: {output_ppt_path}")

if __name__ == "__main__":
    update_presentation() 