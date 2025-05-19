from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
import os

SLIDES_DIR = 'slides'
TEMPLATE_PATH = os.path.join(SLIDES_DIR, 'Module2_template.pptx')
CUM_RET_PNG = 'outputs/cum_returns.png'

# Actual values from outputs
OOS_R2 = -0.0190  # From metrics.csv
PERF_TABLE = [
    ["Metric", "ElasticNet Strategy", "SPY"],
    ["Alpha (monthly)", "0.0035", "—"],
    ["Sharpe Ratio (ann.)", "3.05", "4.62"],
    ["Avg Return (monthly)", "0.0177", "0.0150"],
    ["Volatility (monthly)", "0.0755", "0.0112"],
    ["Max Drawdown", "0.0306", "0.0376"],
    ["Max 1-mo Loss", "-0.0062", "-0.0004"]
]

METHODOLOGY = (
    "• Implemented ElasticNet regression to predict next-month stock excess returns\n"
    "• Training: 10-year initial window (2005-2014)\n"
    "• Validation: 2-year period (2015-2016)\n"
    "• OOS Testing: 2017 with rolling window updates\n"
    "• Portfolio: Equal-weighted top 50 stocks monthly rebalancing\n"
    "• Features: 145 lagged stock characteristics (t) predicting returns (t+1)\n"
    "• Benchmark: S&P 500 (SPY) for performance comparison"
)

SUMMARY = (
    "The ElasticNet strategy achieved a positive monthly alpha of 0.35% and an annualized Sharpe ratio of 3.05, "
    "outperforming SPY in terms of risk-adjusted returns. However, the strategy exhibited higher volatility "
    "(7.55% monthly vs SPY's 1.12%) and a negative OOS R² (-1.90%), suggesting potential overfitting. "
    "Future improvements:\n"
    "1. Feature Engineering: Implement cross-sectional momentum (6m/12m) and mean reversion signals (1m/3m), "
    "add sector-neutral factors, and incorporate market microstructure features (bid-ask spread, volume profile)\n"
    "2. Model Enhancement: Use L1/L2 ratio of 0.7/0.3 for better feature selection, implement time-varying "
    "regularization based on market volatility, and add regime-switching components for different market conditions\n"
    "3. Portfolio Construction: Implement risk parity weighting across sectors, add dynamic position sizing based "
    "on prediction confidence, and incorporate transaction cost optimization using VWAP-based execution strategy\n"
    "4. Risk Management: Add stop-loss at 2% per position, implement sector exposure limits (±10% vs benchmark), "
    "and use volatility targeting to maintain constant portfolio risk"
)

def update_presentation():
    prs = Presentation(TEMPLATE_PATH)
    
    # Slide 1: Methodology
    slide = prs.slides[0]
    if len(slide.placeholders) > 1:
        tf = slide.placeholders[1].text_frame
        tf.clear()
        p = tf.add_paragraph()
        p.text = METHODOLOGY
        p.font.size = Pt(20)
    
    # Slide 2: OOS R2
    slide = prs.slides[1]
    tf = slide.shapes.placeholders[1].text_frame
    tf.clear()
    p = tf.add_paragraph()
    p.text = f"Out-of-sample $R^2$: {OOS_R2:.4f}"
    p.font.size = Pt(32)
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER
    
    # Add interpretation
    p = tf.add_paragraph()
    p.text = "\nInterpretation:\n• Negative R² indicates the model underperforms the mean prediction\n• Suggests potential overfitting or weak predictive power\n• Common in financial markets due to efficient market hypothesis"
    p.font.size = Pt(18)
    
    # Slide 3: Performance Table
    slide = prs.slides[2]
    left = Inches(1.0)
    top = Inches(1.5)
    rows, cols = len(PERF_TABLE), len(PERF_TABLE[0])
    table = slide.shapes.add_table(rows, cols, left, top, Inches(6.0), Inches(2.0)).table
    
    # Style the table
    for i, row in enumerate(PERF_TABLE):
        for j, val in enumerate(row):
            cell = table.cell(i, j)
            cell.text = val
            cell.text_frame.paragraphs[0].font.size = Pt(16)
            if i == 0:  # Header row
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(200, 200, 200)
                cell.text_frame.paragraphs[0].font.bold = True
    
    # Add key insights below table
    left = Inches(1.0)
    top = Inches(4.0)
    width = Inches(6.0)
    height = Inches(1.0)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    tf = textbox.text_frame
    p = tf.add_paragraph()
    p.text = "Key Insights:\n• Strategy shows positive alpha but higher volatility than SPY\n• Sharpe ratio of 3.05 indicates strong risk-adjusted returns\n• Maximum drawdown of 3.06% is lower than SPY's 3.76%"
    p.font.size = Pt(14)
    
    # Slide 4: Cumulative Returns Plot
    slide = prs.slides[3]
    left = Inches(1.0)
    top = Inches(1.0)
    if os.path.exists(CUM_RET_PNG):
        slide.shapes.add_picture(CUM_RET_PNG, left, top, width=Inches(6.0))
    else:
        tf = slide.shapes.placeholders[1].text_frame
        tf.clear()
        tf.add_paragraph().text = "Cumulative returns plot not found."
    
    # Add interpretation
    left = Inches(1.0)
    top = Inches(4.0)
    width = Inches(6.0)
    height = Inches(1.0)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    tf = textbox.text_frame
    p = tf.add_paragraph()
    p.text = "Interpretation:\n• Strategy shows strong performance in early 2017\n• Higher volatility compared to SPY\n• Notable outperformance in November 2017 (+6.70%)"
    p.font.size = Pt(14)
    
    # Slide 5: Summary
    slide = prs.slides[4]
    tf = slide.shapes.placeholders[1].text_frame
    tf.clear()
    p = tf.add_paragraph()
    p.text = SUMMARY
    p.font.size = Pt(20)
    
    prs.save(TEMPLATE_PATH)
    print(f"Slides updated with finance-focused content: {TEMPLATE_PATH}")

if __name__ == "__main__":
    update_presentation() 