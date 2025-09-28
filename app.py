import os
import streamlit as st

# Read GEMINI_API_KEY from Streamlit Secrets if present (for Cloud)
if "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]



import tempfile
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None



from google import genai
client = genai.Client()  # picks up GEMINI_API_KEY from your environment

import json, os, datetime
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Greenlight Home — Pre-Qual MVP', page_icon='✅', layout='wide')

# ----------------------- Helpers -----------------------
@st.cache_data
def load_programs(path: str = 'programs.json') -> pd.DataFrame:
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame(columns=['state','name','summary','url','notes','tags'])

def amortization_payment(principal: float, annual_rate: float, years: int) -> float:
    '''Monthly principal+interest for fixed-rate fully-amortizing loan.'''
    if principal <= 0 or annual_rate < 0 or years <= 0:
        return 0.0
    r = annual_rate / 12.0
    n = years * 12
    if r == 0:
        return principal / n
    return principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)

def pmi_rate_annual(ltv: float, credit_band: str) -> float:
    '''Very simple PMI approximation. Production apps should use live rate cards.'''
    base = 0.0
    if ltv >= 0.97:
        base = 0.012
    elif ltv >= 0.95:
        base = 0.010
    elif ltv >= 0.90:
        base = 0.008
    elif ltv >= 0.85:
        base = 0.006
    elif ltv >= 0.80:
        base = 0.004
    else:
        base = 0.0
    mult = {'Excellent':0.8, 'Good':1.0, 'Fair':1.2, 'Poor':1.4}.get(credit_band, 1.0)
    return base * mult

def piti_breakdown(price: float, dp_amount: float, rate: float, term_years: int,
                   tax_rate: float, ins_rate: float, hoa_mo: float, credit_band: str,
                   loan_type: str = "Conventional", rules: dict | None = None) -> dict:
    '''Compute monthly PITI and components.
    For FHA/VA/USDA, we:
      - finance the upfront fee into the loan (MIP/Funding/Guarantee fee),
      - use an annual MIP if present instead of conventional PMI.
    '''
    price = max(0.0, price)
    dp_amount = max(0.0, dp_amount)
    base_loan = max(0.0, price - dp_amount)

    # Finance upfront fee (rough demo assumption)
    loan = float(base_loan)
    if rules and rules.get("mip_upfront", 0) > 0:
        loan = loan * (1.0 + float(rules["mip_upfront"]))

    pi = amortization_payment(loan, rate, term_years)
    taxes = (price * tax_rate) / 12.0
    ins = (price * ins_rate) / 12.0

    ltv = (loan / price) if price > 0 else 0.0

    # Monthly insurance
    pmi = 0.0
    if loan_type == "Conventional":
        if ltv >= 0.80:
            pmi = (loan * pmi_rate_annual(ltv, credit_band)) / 12.0
    else:
        mip_annual = (rules or {}).get("mip_annual", 0.0) or 0.0
        if mip_annual and loan > 0:
            pmi = (loan * mip_annual) / 12.0

    piti = pi + taxes + ins + pmi + hoa_mo
    return {"PI": pi, "Taxes": taxes, "Insurance": ins, "PMI": pmi, "HOA": hoa_mo,
            "PITI": piti, "LTV": ltv, "Loan": loan, "BaseLoan": base_loan}

def affordable_price_binary(target_piti: float, dp_cash: float, rate: float, term_years: int,
                            tax_rate: float, ins_rate: float, hoa_mo: float, credit_band: str,
                            loan_type: str, rules: dict | None,
                            lo: float = 50000.0, hi: float = 2000000.0, tol: float = 1.0) -> float:
    '''Binary search a home price whose PITI is ~ target (given fixed down-payment cash).'''
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        b = piti_breakdown(mid, dp_cash, rate, term_years, tax_rate, ins_rate, hoa_mo, credit_band, loan_type, rules)
        if b["PITI"] > target_piti:
            hi = mid
        else:
            lo = mid
        if abs(hi - lo) < tol:
            break
    return max(lo, 0.0)

def next_steps(copy: dict) -> list:
    tips = []
    if copy['dp_percent'] < 0.20 and copy['price_likely'] > 0:
        need = max(0.0, copy['price_likely'] * 0.20 - copy['dp_cash'])
        if need > 0:
            tips.append(f'Raising down payment by approximately ${need:,.0f} would remove PMI and lower PITI.')
    if copy['income_mo'] > 0 and copy['dti_back_likely'] > copy['dti_back_threshold_likely']:
        max_housing = copy['dti_back_threshold_likely'] * copy['income_mo']
        need_drop = max(0.0, (copy['debts_mo'] + copy['piti_likely']) - max_housing)
        if need_drop > 0:
            tips.append(f'Paying down about ${need_drop:,.0f} in monthly obligations (or refinancing to reduce payments) would meet the DTI threshold.')
    if copy['credit_band'] in ['Fair','Poor'] and copy['dp_percent'] < 0.20:
        tips.append('Improving your credit band could reduce PMI and rate; consider on-time payments and lowering credit utilization.')
    if copy['hoa_mo'] > 0:
        tips.append('Consider homes with lower or no HOA fees to expand your affordable range.')
    if len(tips) == 0:
        tips.append('You appear ready. Consider getting a soft-pull pre-qualification and contacting a mortgage broker.')
    return tips[:5]

def badge_emoji(level: str) -> str:
    return {'Green':'✅', 'Amber':'🟡', 'Red':'🔴'}.get(level, '⚪')

# ----------------------- Sidebar -----------------------
st.sidebar.header('Calculator Settings')
st.sidebar.caption('Tune the defaults to your market.')

front_conservative = st.sidebar.slider('Front-end DTI (Conservative)', 0.20, 0.35, 0.28, 0.01)
back_conservative  = st.sidebar.slider('Back-end DTI (Conservative)', 0.25, 0.50, 0.36, 0.01)
front_likely_user  = st.sidebar.slider('Front-end DTI (Likely, base)', 0.20, 0.40, 0.31, 0.01)
back_likely_user   = st.sidebar.slider('Back-end DTI (Likely, base)', 0.30, 0.55, 0.43, 0.01)

default_tax_rate   = st.sidebar.number_input('Default annual property tax rate (%)', 0.0, 5.0, 1.25, 0.05) / 100.0
default_ins_rate   = st.sidebar.number_input('Default annual homeowner insurance rate (%)', 0.0, 2.0, 0.50, 0.05) / 100.0

st.sidebar.divider()
st.sidebar.caption('These are heuristic defaults for the MVP. Actual underwriting varies by lender/program.')

# ----------------------- Main -----------------------
st.title('Greenlight Home — Pre‑Qualification MVP')
st.write('Transparent, no‑hard‑pull estimate of what you can afford, plus clear next steps.')

col1, col2, col3 = st.columns([1.0, 1.0, 1.4], gap='large')

with col1:
    st.subheader('Location & Profile')
    state = st.selectbox('State', ['PA','NJ','NY','DE','MD','Other'], index=0)
    city_zip = st.text_input('City or ZIP (optional)', placeholder='e.g., Philadelphia 19130')
    credit_band = st.selectbox('Credit band (self‑reported)', ['Excellent','Good','Fair','Poor'], index=1)

    st.subheader('Income & Debts')
    income_mo = st.number_input('Gross household income per month ($)', min_value=0.0, value=9000.0, step=500.0, format='%.2f')
    debts_mo  = st.number_input('Total monthly debt payments ($)', min_value=0.0, value=800.0, step=50.0, format='%.2f')

with col2:
    st.subheader('Home & Loan')
    # (1) Loan type & rules
    loan_type = st.selectbox("Loan type", ["Conventional", "FHA", "VA", "USDA"], index=0)
    RULES = {
        "Conventional": {"front": (0.28, 0.31), "back": (0.36, 0.43), "min_dp": 0.03,  "mip_upfront": 0.0,   "mip_annual": None},
        "FHA":          {"front": (0.31, 0.31), "back": (0.43, 0.43), "min_dp": 0.035, "mip_upfront": 0.0175,"mip_annual": 0.0055},
        "VA":           {"front": (0.31, 0.31), "back": (0.41, 0.41), "min_dp": 0.0,   "mip_upfront": 0.0225,"mip_annual": 0.0},
        "USDA":         {"front": (0.29, 0.29), "back": (0.41, 0.41), "min_dp": 0.0,   "mip_upfront": 0.01,  "mip_annual": 0.0035},
    }
    rules = RULES[loan_type]

    rate = st.number_input('Annual interest rate (%)', 0.0, 15.0, 6.8, 0.05) / 100.0
    term = st.selectbox('Loan term (years)', [15, 20, 30], index=2)
    hoa_mo = st.number_input('HOA dues per month ($)', 0.0, 1500.0, 0.0, 10.0, format='%.2f')

    # --- State-based default tax/insurance overrides ---
    STATE_DEFAULTS = {
        "PA": {"tax": 1.35, "ins": 0.45},
        "NJ": {"tax": 2.20, "ins": 0.50},
        "NY": {"tax": 1.70, "ins": 0.55},
        "DE": {"tax": 0.60, "ins": 0.45},
        "MD": {"tax": 1.10, "ins": 0.55},
    }
    state_tax_pct = STATE_DEFAULTS.get(state, {}).get("tax", default_tax_rate*100.0)
    state_ins_pct = STATE_DEFAULTS.get(state, {}).get("ins", default_ins_rate*100.0)
    tax_rate = state_tax_pct / 100.0
    ins_rate = state_ins_pct / 100.0
    st.caption(f"Using state defaults — Taxes: {state_tax_pct:.2f}% • Insurance: {state_ins_pct:.2f}%  (edit in sidebar if needed)")

    st.subheader('Down Payment')
    dp_cash = st.number_input('Down payment cash available ($)', 0.0, 500000.0, 30000.0, 1000.0, format='%.2f')
    gift_funds = st.number_input('Gift funds (optional) ($)', 0.0, 500000.0, 0.0, 1000.0, format='%.2f')
    dp_cash_total = dp_cash + gift_funds

with col3:
    st.subheader('Target ranges')
    st.caption('We’ll solve for the maximum price that keeps PITI within DTI caps.')

    # Loan-type-adjusted likely caps; conservative remains as sidebar
    front_likely = rules["front"][1] if rules else front_likely_user
    back_likely  = rules["back"][1]  if rules else back_likely_user

    # Max PITI from DTI thresholds
    piti_cap_front_cons = front_conservative * income_mo
    piti_cap_back_cons  = max(0.0, back_conservative * income_mo - debts_mo)
    piti_cap_front_lik  = front_likely * income_mo
    piti_cap_back_lik   = max(0.0, back_likely * income_mo - debts_mo)

    cons_target = min(piti_cap_front_cons, piti_cap_back_cons)
    lik_target  = min(piti_cap_front_lik, piti_cap_back_lik)

    # Solve price bands with loan type rules
    price_cons = affordable_price_binary(cons_target, dp_cash_total, rate, term, tax_rate, ins_rate, hoa_mo, credit_band, loan_type, rules)
    price_lik  = affordable_price_binary(lik_target,  dp_cash_total, rate, term, tax_rate, ins_rate, hoa_mo, credit_band, loan_type, rules)
    price_str  = f'${price_cons:,.0f} – ${price_lik:,.0f}'

    # Compute likely breakdown for display
    bd = piti_breakdown(price_lik, dp_cash_total, rate, term, tax_rate, ins_rate, hoa_mo, credit_band, loan_type, rules)
    dti_front = bd['PITI'] / income_mo if income_mo > 0 else 0.0
    dti_back  = (bd['PITI'] + debts_mo) / income_mo if income_mo > 0 else 0.0

    # Eligibility badge (use loan-type min DP)
    level = 'Red'
    dp_pct_at_likely = (dp_cash_total / price_lik) if price_lik > 0 else 0.0
    if price_lik > 0 and dti_back <= back_likely and dti_front <= front_likely and dp_pct_at_likely >= rules["min_dp"]:
        level = 'Green'
    elif price_lik > 0 and (dti_back <= (back_likely + 0.02) or dti_front <= (front_likely + 0.02)):
        level = 'Amber'

    st.metric('Affordability range (Conservative → Likely)', price_str)
    st.metric('Estimated monthly PITI at Likely price', f'${bd["PITI"]:,.0f}')
    st.metric('Eligibility', f'{level} {badge_emoji(level)}')

    with st.expander('PITI breakdown @ Likely price'):
        show_bd = {k: f'${v:,.0f}' for k,v in bd.items() if k in ['PI','Taxes','Insurance','PMI','HOA','PITI']}
        st.write(show_bd)

    with st.expander('DTI snapshot @ Likely price'):
        st.write({
            'Front-end DTI': f'{dti_front*100:.1f}% (cap {front_likely*100:.0f}%)',
            'Back-end DTI': f'{dti_back*100:.1f}% (cap {back_likely*100:.0f}%)'
        })

    # --- Cash-to-close & closing costs ---
    st.subheader("Cash to Close")
    closing_pct = st.slider("Estimate closing costs (% of price)", 1.5, 5.0, 3.0, 0.25) / 100.0
    closing_costs = price_lik * closing_pct
    out_of_pocket = max(0.0, (dp_cash + closing_costs) - gift_funds)
    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Closing costs (est.)", f"${closing_costs:,.0f}")
    with colB:
        st.metric("Cash to close (you)", f"${out_of_pocket:,.0f}")
    with colC:
        st.metric("Down payment % @ Likely", f"{(dp_pct_at_likely*100):.1f}%")

    # --- What-if sensitivity ---
    st.subheader("What‑if: rate & down‑payment")
    r1, r2 = st.columns(2)
    with r1:
        delta_rate = st.slider("Rate change (±%)", -1.0, 1.0, 0.5, 0.05) / 100.0
        alt_rate = max(0.0, rate + delta_rate)
        alt_price = affordable_price_binary(lik_target, dp_cash_total, alt_rate, term, tax_rate, ins_rate, hoa_mo, credit_band, loan_type, rules)
        st.write(f"At **{(alt_rate*100):.2f}%**, likely price ≈ **${alt_price:,.0f}**")
    with r2:
        extra_dp = st.slider("Extra down payment ($)", 0, 50000, 10000, 1000)
        alt_price2 = affordable_price_binary(lik_target, dp_cash_total + extra_dp, rate, term, tax_rate, ins_rate, hoa_mo, credit_band, loan_type, rules)
        st.write(f"With **+${extra_dp:,}** DP, likely price ≈ **${alt_price2:,.0f}**")

# ----------------------- Next steps & Programs -----------------------
dp_percent = (dp_cash_total / price_lik) if price_lik > 0 else 0.0
ctx = {
    'dp_percent': dp_percent,
    'price_likely': price_lik,
    'dp_cash': dp_cash_total,
    'dti_back_likely': (bd['PITI'] + debts_mo) / income_mo if income_mo > 0 else 0.0,
    'dti_back_threshold_likely': back_likely,
    'income_mo': income_mo,
    'debts_mo': debts_mo,
    'piti_likely': bd['PITI'],
    'credit_band': credit_band,
    'hoa_mo': hoa_mo
}
st.subheader('Next steps')
for tip in next_steps(ctx):
    st.write('• ' + tip)

# ----------------------- Program filters & listing -----------------------
st.subheader('Programs near you (sample)')
df = load_programs()
programs_for_plan = pd.DataFrame()
if not df.empty:
    # Filter controls
    st.caption("Filter programs:")
    fc1, fc2, fc3 = st.columns(3)
    want_ftb = fc1.checkbox("First‑time buyer", value=False)
    want_dpa = fc2.checkbox("Down‑payment help (DPA)", value=False)
    want_credit = fc3.checkbox("Credit‑builder", value=False)

    show = df[df['state'] == state]
    if show.empty:
        st.caption('No curated programs for your state yet. Showing nearby examples:')
        show = df[df['state'].isin(['PA','NJ'])]

    # Apply tag filter if tags exist
    if "tags" in show.columns and (want_ftb or want_dpa or want_credit):
        def row_tag_match(r):
            tags = r["tags"] if "tags" in r and isinstance(r["tags"], (list, set, tuple)) else []
            tags = set(tags)
            if want_ftb and "first-time" not in tags: return False
            if want_dpa and "DPA" not in tags: return False
            if want_credit and "credit-builder" not in tags: return False
            return True
        show = show[show.apply(row_tag_match, axis=1)]

    programs_for_plan = show.copy()
    if show.empty:
        st.caption("No programs match those filters. Try broadening your selection.")
    for _, r in show.iterrows():
        url = r.get('url', '')
        notes = r.get('notes', '')
        summary = r.get('summary', '')
        st.markdown(f"**{r['name']}** — {summary}  \\n{notes}  " + (f"[Learn more ↗]({url})" if url else ""))
else:
    st.caption('No program data file found.')

# ----------------------- Lead capture & downloadable plan -----------------------
st.subheader('Get your plan by email (demo)')
with st.form('lead_capture'):
    name  = st.text_input('Your name')
    email = st.text_input('Email')
    consent = st.checkbox('I agree to be contacted about mortgage/agent options (demo).')
    submitted = st.form_submit_button('Email my plan')

if submitted:
    if not email or not consent:
        st.error('Please provide an email and consent to be contacted.')
    else:
        # Build simple text plan
        lines = [
            'Greenlight Home — Your Plan',
            f'Date: {datetime.date.today():%B %d, %Y}',
            '',
            f'Loan type: {loan_type}',
            f'Affordability range: ${price_cons:,.0f} – ${price_lik:,.0f}',
            f"Estimated monthly PITI (Likely): ${bd['PITI']:,.0f}",
            f'Front-end DTI: {dti_front*100:.1f}%   Back-end DTI: {dti_back*100:.1f}%',
            f'Closing costs (est.): ${closing_costs:,.0f}   Cash to close (you): ${out_of_pocket:,.0f}',
            '',
            'Next steps:'
        ]
        for tip in next_steps(ctx):
            lines.append(f'- {tip}')
        lines.append('')
        if not programs_for_plan.empty:
            lines.append('Programs (sample):')
            for _, r in programs_for_plan.iterrows():
                nm = r.get('name','Program')
                url = r.get('url','')
                lines.append(f"- {nm}" + (f" — {url}" if url else ""))
        plan_txt = "\n".join(lines)

        # Save a simple lead row locally (demo)
        row = {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "name": name, "email": email, "state": state, "loan_type": loan_type,
            "price_cons": round(price_cons), "price_lik": round(price_lik),
            "piti": round(bd["PITI"]), "dti_front": round(dti_front,4), "dti_back": round(dti_back,4),
            "closing_costs": round(closing_costs), "cash_to_close_you": round(out_of_pocket)
        }
        leads_path = "leads.csv"
        try:
            if os.path.exists(leads_path):
                pd.concat([pd.read_csv(leads_path), pd.DataFrame([row])]).to_csv(leads_path, index=False)
            else:
                pd.DataFrame([row]).to_csv(leads_path, index=False)
        except Exception:
            pd.DataFrame([row]).to_csv(leads_path, index=False)

        st.success('Plan ready! Download it below. (Email send is simulated for the demo.)')
        st.download_button('Download my plan (.txt)', plan_txt, file_name='greenlight_plan.txt')

# ----------------------- Admin (demo) -----------------------
with st.expander("Admin (demo): leads & maintenance"):
    if os.path.exists("leads.csv"):
        df_leads = pd.read_csv("leads.csv")
        st.dataframe(df_leads, use_container_width=True)
        st.download_button("Download leads.csv", df_leads.to_csv(index=False), file_name="leads.csv")
    else:
        st.caption("No leads yet.")
    if st.button("Reset leads (delete file)"):
        try:
            os.remove("leads.csv")
            st.success("Leads reset.")
        except Exception as e:
            st.error(f"Could not delete leads.csv: {e}")

st.divider()
st.caption('This tool is for education and planning. It is not a credit decision or a commitment to lend. Fees, PMI/MIP, taxes and insurance are estimates only and vary by lender, program, and property.')


st.subheader("Upload a pay stub or Loan Estimate (PDF) — AI summary")

uploaded = st.file_uploader("Choose a PDF to analyze", type=["pdf"])

if uploaded is not None:
    # 1) Save the uploaded file to a temporary path (Files API needs a path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name

    # 2) Upload file to Gemini Files API and ask for a summary
    with st.spinner("Analyzing your document…"):
        try:
            myfile = client.files.upload(file=tmp_path)

            doc_prompt = (
                "You are a careful mortgage coach. Read this document and extract "
                "mortgage-relevant details. If present, include: gross MONTHLY income, "
                "other income, deductions, escrow items, interest rate/APR, term, points/fees, "
                "and any red flags for underwriting. Return 5–8 concise bullets plus a one-line summary."
            )

            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[doc_prompt, myfile]
            )

            st.success("AI summary")
            st.write(resp.text)

        except Exception as e:
            st.error(f"AI document analysis failed: {e}")
            st.caption("Tip: ensure GEMINI_API_KEY is set and try a smaller or text-based PDF.")

    # 3) Optional: show raw extracted text (if the PDF has text; scanned images may be blank)
    if PdfReader is not None:
        try:
            text = ""
            reader = PdfReader(tmp_path)
            for p in reader.pages:
                text += p.extract_text() or ""
            with st.expander("Show raw text (from PDF)"):
                st.write(text[:5000] + ("…" if len(text) > 5000 else ""))
        except Exception:
            pass





st.subheader("Ask the Homebuying Coach (AI)")

q = st.text_input("Ask about DTI, FHA vs Conventional, PMI/MIP, closing costs, etc.")
if q:
    try:
        dp_total = (dp_cash + gift_funds) if 'dp_cash' in locals() and 'gift_funds' in locals() else 0
        prompt = f"""
You are a careful, plain-English mortgage coach.
User profile (if helpful): state={state}, income_mo={income_mo}, debts_mo={debts_mo},
credit_band={credit_band}, down_payment_total={dp_total}.
Question: {q}
Please answer in 3–6 short bullets and include one realistic caution or next step.
"""
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        st.write(resp.text)
    except Exception as e:
        st.error(f"AI error: {e}")
        st.caption("Tip: make sure GEMINI_API_KEY is set in this terminal before running Streamlit.")
