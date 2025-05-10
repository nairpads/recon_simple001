import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🔍 Ledger to Bank Statement Reconciliation")

# Upload files
bank_file = st.file_uploader("Upload Bank Statement CSV", type=["csv"])
ledger_file = st.file_uploader("Upload Ledger Entries CSV", type=["csv"])

if bank_file and ledger_file:
    # Load CSVs
    bank_df = pd.read_csv(bank_file)
    ledger_df = pd.read_csv(ledger_file)

    # Smart column detection for bank description
    possible_bank_cols = ['description', 'transaction details', 'details', 'narrative']
    bank_desc_col = next((col for col in bank_df.columns if col.strip().lower() in possible_bank_cols), None)

    if not bank_desc_col:
        st.error("Could not find a suitable 'description' column in bank statement file.")
        st.stop()

    # Smart column detection for ledger narration
    possible_ledger_cols = ['narration', 'description', 'details', 'remarks']
    ledger_desc_col = next((col for col in ledger_df.columns if col.strip().lower() in possible_ledger_cols), None)

    if not ledger_desc_col:
        st.error("Could not find a suitable 'narration' column in ledger file.")
        st.stop()

    # Normalize text
    bank_df['description'] = bank_df[bank_desc_col].astype(str).str.lower()
    ledger_df['narration'] = ledger_df[ledger_desc_col].astype(str).str.lower()

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    vectorizer.fit(bank_df['description'].tolist() + ledger_df['narration'].tolist())

    bank_tfidf = vectorizer.transform(bank_df['description'])
    ledger_tfidf = vectorizer.transform(ledger_df['narration'])

    # Cosine Similarity
    similarity_matrix = cosine_similarity(bank_tfidf, ledger_tfidf)

    # Best Matches
    best_matches = similarity_matrix.argmax(axis=1)
    match_scores = similarity_matrix.max(axis=1)

    # Append match results
    bank_df['matched_ledger_index'] = best_matches
    bank_df['similarity_score'] = match_scores
    bank_df['matched_ledger_date'] = ledger_df.loc[best_matches, 'date'].values
    bank_df['matched_ledger_amount'] = ledger_df.loc[best_matches, 'amount'].values
    bank_df['matched_ledger_narration'] = ledger_df.loc[best_matches, 'narration'].values
    bank_df['match_confidence'] = bank_df['similarity_score'].apply(lambda x: 'High' if x > 0.75 else 'Low')

    # Show result
    st.success("✅ Matching complete!")
    st.dataframe(bank_df)

    # Download button
    csv = bank_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Reconciled Results",
        data=csv,
        file_name="reconciled_output.csv",
        mime="text/csv",
    )

