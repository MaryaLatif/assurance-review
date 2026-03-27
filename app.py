import streamlit as st
import torch
import pandas as pd
import numpy as np
from predict import load_model, predict_review

st.set_page_config(
    page_title="Analyse d'avis assurance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=80)
    st.title("🛡️ InsurReview AI")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Accueil",
            "🔮 Prédiction",
            "📊 Résumé",
            "💡 Explication",
            "🔍 Information Retrieval"
        ]
    )

    st.markdown("---")
    st.markdown("### 📌 À propos")
    st.markdown("""
- **Dataset** : 24 000+ avis assurance  
- **Langue** : Anglais (traduit)  
- **Meilleur modèle** : DistilBERT  
- **Accuracy** : 51.67%  
    """)
    st.markdown("---")
    st.caption("Projet NLP — Supervised Learning")

# ─────────────────────────────────────────────
# Chargement des ressources
# ─────────────────────────────────────────────
@st.cache_resource
def init_model():
    return load_model()

@st.cache_data
def load_data():
    df = pd.read_csv("datas/data_supervised.csv", index_col=0)
    df["note"] = df["note"].astype(int)
    return df

@st.cache_resource
def init_tfidf(df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(df["avis_en"].fillna(""))
    matrix_norm = normalize(matrix)
    return vectorizer, matrix_norm

tokenizer, model, is_finetuned = init_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
if not is_finetuned:
    st.sidebar.warning("⚠️ Modèle de base chargé — les prédictions sont aléatoires.")

df = load_data()
vectorizer, tfidf_matrix = init_tfidf(df)

# ══════════════════════════════════════════════
# PAGE 0 — ACCUEIL
# ══════════════════════════════════════════════
if page == "🏠 Accueil":
    st.title("🛡️ InsurReview AI")
    st.subheader("Analyse intelligente d'avis d'assurance par NLP")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
## 📖 Description du projet

Ce projet applique des techniques de **traitement du langage naturel (NLP)** 
sur un dataset de **plus de 24 000 avis clients** issus de compagnies d'assurance françaises.

L'objectif principal est de **prédire automatiquement la note** (1 à 5 étoiles) 
associée à un avis, à partir du texte seul.

### 🔬 Modèles comparés

| Modèle | Accuracy |
|---|---|
| TF-IDF + Logistic Regression | 51.07% |
| Embedding PyTorch (from scratch) | ~48% |
| Word2Vec + Logistic Regression | ~49% |
| **DistilBERT fine-tuné** | **51.67%** ✅ |

Le modèle retenu est **DistilBERT** (`distilbert-base-uncased`), 
fine-tuné sur 3 epochs avec un GPU T4 (Google Colab).

### 📂 Dataset

- **Source** : Avis clients scrappés sur des plateformes d'assurance françaises  
- **Taille** : 24 102 avis après nettoyage  
- **Langue** : Français traduit en anglais  
- **Classes** : Notes de 1 à 5 étoiles (classification multi-classe)  
- **Distribution** : Légèrement déséquilibrée (surreprésentation des notes 1 et 5)
        """)

    with col2:
        st.markdown("### 🚀 Fonctionnalités")
        st.markdown("""
**🔮 Prediction**  
Prédit la note d'un avis avec DistilBERT

---

**📊 Summary**  
Statistiques générales du dataset et visualisations

---

**💡 Explanation**  
Identifie les mots les plus influents dans la prédiction via TF-IDF

---

**🔍 Information Retrieval**  
Retrouve les avis les plus similaires à une requête via cosine similarity
        """)

        st.markdown("---")
        st.markdown("### ⚙️ Stack technique")
        st.markdown("""
- 🤗 HuggingFace Transformers  
- 🔥 PyTorch  
- 📊 Scikit-learn  
- 🌊 Streamlit  
- 🐼 Pandas / NumPy  
        """)

    st.markdown("---")
    st.info("👈 Utilise la **sidebar gauche** pour naviguer entre les sections.")

# ══════════════════════════════════════════════
# PAGE 1 — PREDICTION
# ══════════════════════════════════════════════
elif page == "🔮 Prédiction":
    st.title("🔮 Prédiction de note")
    st.caption("Modèle DistilBERT fine-tuné — accuracy 51.67% sur le jeu de test")
    st.markdown("---")

    if "review_text" not in st.session_state:
        st.session_state.review_text = ""

    examples = [
        "Best insurance, great price and fast service",
        "Price increase without reason, impossible to reach",
        "Correct price but claim badly handled",
        "Generally satisfied, easy renewal process",
        "Worst experience ever, avoid this company"
    ]

    col1, col2 = st.columns([3, 1])

    with col2:
        st.markdown("### 💡 Exemples")
        selected = st.selectbox("Choisir un exemple", examples)
        if st.button("Utiliser cet exemple"):
            st.session_state.review_text = selected

    with col1:
        # PAS de key= ici — sinon Streamlit ignore value=
        review = st.text_area(
            "Saisis un avis (en anglais)",
            value=st.session_state.review_text,
            placeholder="Ex: 'Best insurance, great price and fast service'...",
            height=150,
        )

    if st.button("🔮 Prédire la note", type="primary"):
        if review.strip():
            with st.spinner("Prédiction en cours..."):
                result = predict_review(review, tokenizer, model, device)

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Note prédite", f"{result['prediction']} ⭐")
            with col_b:
                st.metric("Confiance", f"{result['confidence']:.1%}")

            probs_df = pd.DataFrame([
                {"Note": k, "Probabilité": round(v, 4)}
                for k, v in result["probabilities"].items()
            ]).set_index("Note")
            st.bar_chart(probs_df)
        else:
            st.warning("⚠️ Saisis un avis pour pouvoir prédire !")

# ══════════════════════════════════════════════
# PAGE 2 — SUMMARY
# ══════════════════════════════════════════════
elif page == "📊 Résumé":
    st.title("📊 Résumé du dataset")
    st.markdown("---")

    import plotly.express as px

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total avis", f"{len(df):,}")
    col2.metric("Note moyenne", f"{df['note'].mean():.2f} ⭐")
    col3.metric("Assureurs uniques", df["assureur"].nunique() if "assureur" in df.columns else "—")
    col4.metric("Longueur moyenne", f"{df['avis_en'].fillna('').apply(lambda x: len(x.split())).mean():.0f} mots")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Distribution des notes")
        note_counts = df["note"].value_counts().sort_index().reset_index()
        note_counts.columns = ["Note", "Nombre d'avis"]
        note_counts["Note"] = note_counts["Note"].astype(str) + " ⭐"
        fig1 = px.bar(note_counts, x="Note", y="Nombre d'avis", color="Nombre d'avis",
                      color_continuous_scale="blues")
        fig1.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        st.subheader("Top 10 assureurs")
        if "assureur" in df.columns:
            top_assureurs = df["assureur"].value_counts().head(10).reset_index()
            top_assureurs.columns = ["Assureur", "Nombre d'avis"]
            fig2 = px.bar(top_assureurs, x="Nombre d'avis", y="Assureur",
                          orientation="h", color="Nombre d'avis",
                          color_continuous_scale="teal")
            fig2.update_layout(yaxis={"categoryorder": "total ascending"},
                               coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Colonne 'assureur' non disponible.")

    st.markdown("---")
    st.subheader("Note moyenne par assureur (min. 20 avis)")
    if "assureur" in df.columns:
        avg_note = df.groupby("assureur")["note"].agg(["mean", "count"])
        avg_note = avg_note[avg_note["count"] >= 20].sort_values("mean", ascending=False).head(10).reset_index()
        avg_note.columns = ["Assureur", "Note moyenne", "Nombre d'avis"]
        fig3 = px.bar(avg_note, x="Note moyenne", y="Assureur",
                      orientation="h", color="Note moyenne",
                      color_continuous_scale="RdYlGn", range_color=[1, 5])
        fig3.update_layout(yaxis={"categoryorder": "total ascending"},
                           coloraxis_showscale=True)
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════
# PAGE 3 — EXPLANATION
# ══════════════════════════════════════════════
elif page == "💡 Explication":
    st.title("💡 Explication de la prédiction")
    st.caption("Mots les plus influents + probabilités détaillées par classe")
    st.markdown("---")

    review_exp = st.text_area(
        "Saisis un avis à expliquer (en anglais)",
        placeholder="Ex: 'Terrible service, no response, avoid this company'...",
        height=120,
        key="explain_input"
    )

    top_n = st.slider("Nombre de mots à afficher", min_value=5, max_value=20, value=10)

    if st.button("💡 Expliquer", type="primary"):
        if review_exp.strip():
            result = predict_review(review_exp, tokenizer, model, device)

            # ── Métriques principales
            st.markdown("### 🎯 Résultat de la prédiction")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Note prédite", f"{result['prediction']} ⭐")
            with col_b:
                st.metric("Confiance", f"{result['confidence']:.1%}")
            with col_c:
                sentiment = "😊 Positif" if result['prediction'] >= 4 else ("😐 Neutre" if result['prediction'] == 3 else "😠 Négatif")
                st.metric("Sentiment", sentiment)

            st.markdown("---")

            # ── Probabilités détaillées par classe
            st.markdown("### 📊 Probabilités par classe")
            probs = result["probabilities"]
            probs_df = pd.DataFrame([
                {"Note": k, "Probabilité": round(v, 4)}
                for k, v in probs.items()
            ]).set_index("Note")

            col1, col2 = st.columns([2, 1])
            with col1:
                st.bar_chart(probs_df)
            with col2:
                st.markdown("**Détail :**")
                for note, prob in probs.items():
                    bar = "█" * int(prob * 20)
                    st.markdown(f"`{note}` {bar} **{prob:.1%}**")

            st.markdown("---")

            # ── Mots importants globaux (TF-IDF)
            st.markdown("### 🔑 Mots les plus importants dans cet avis")
            vec = vectorizer.transform([review_exp])
            feature_names = vectorizer.get_feature_names_out()
            scores = vec.toarray()[0]
            top_indices = scores.argsort()[::-1][:top_n]
            top_words = [(feature_names[i], round(scores[i], 4)) for i in top_indices if scores[i] > 0]

            if top_words:
                words_df = pd.DataFrame(top_words, columns=["Mot", "Score TF-IDF"]).set_index("Mot")
                st.bar_chart(words_df)

                # ── Comparaison : mots associés aux avis 1⭐ vs 5⭐ dans le corpus
                st.markdown("---")
                st.markdown("### ⚖️ Ces mots sont-ils plutôt positifs ou négatifs ?")
                st.caption("Comparaison de la note moyenne des avis contenant chaque mot clé")

                word_analysis = []
                for word, score in top_words:
                    # Avis du corpus qui contiennent ce mot
                    mask = df["avis_en"].fillna("").str.contains(r'\b' + word + r'\b', regex=True, case=False)
                    if mask.sum() >= 5:
                        avg = df.loc[mask, "note"].mean()
                        count = mask.sum()
                        word_analysis.append({
                            "Mot": word,
                            "Note moyenne dans le corpus": round(avg, 2),
                            "Nb d'avis": count,
                            "Score TF-IDF": score
                        })

                if word_analysis:
                    analysis_df = pd.DataFrame(word_analysis).set_index("Mot")
                    analysis_df = analysis_df.sort_values("Note moyenne dans le corpus", ascending=False)

                    # Coloriser selon note moyenne
                    def color_note(val):
                        if val >= 4:
                            return "background-color: #d4edda"  # vert
                        elif val <= 2:
                            return "background-color: #f8d7da"  # rouge
                        else:
                            return "background-color: #fff3cd"  # jaune

                    st.dataframe(
                        analysis_df.style.applymap(color_note, subset=["Note moyenne dans le corpus"]),
                        use_container_width=True
                    )
                    st.caption("🟢 Vert = mot associé à des avis positifs | 🔴 Rouge = mot associé à des avis négatifs | 🟡 Jaune = neutre")

            else:
                st.warning("Aucun mot reconnu dans le vocabulaire TF-IDF. Essaie un avis plus long.")

        else:
            st.warning("⚠️ Saisis un avis à expliquer !")

# ══════════════════════════════════════════════
# PAGE 4 — INFORMATION RETRIEVAL
# ══════════════════════════════════════════════
elif page == "🔍 Information Retrieval":
    st.title("🔍 Recherche d'avis similaires")
    st.caption("Retrouve les avis les plus proches de ta requête par cosine similarity TF-IDF")
    st.markdown("---")

    query = st.text_area(
        "Saisis une requête (en anglais)",
        placeholder="Ex: 'slow claim process, no reimbursement'...",
        height=120,
        key="retrieval_input"
    )

    top_k = st.slider("Nombre de résultats", min_value=3, max_value=10, value=5)

    if st.button("🔍 Rechercher", type="primary"):
        if query.strip():
            from sklearn.metrics.pairwise import cosine_similarity
            from sklearn.preprocessing import normalize

            query_vec = vectorizer.transform([query])
            query_vec_norm = normalize(query_vec)
            similarities = cosine_similarity(query_vec_norm, tfidf_matrix)[0]
            top_indices = similarities.argsort()[::-1][:top_k]

            st.subheader(f"📋 Top {top_k} avis similaires")
            for rank, idx in enumerate(top_indices, 1):
                row = df.iloc[idx]
                sim_score = similarities[idx]
                label = f"#{rank} — Note : {row['note']}⭐ — Similarité : {sim_score:.3f}"
                with st.expander(label):
                    if "avis_en" in df.columns:
                        st.write(row["avis_en"])
                    else:
                        st.write(row["textclean"])
                    if "assureur" in df.columns:
                        st.caption(f"Assureur : {row['assureur']}")
        else:
            st.warning("⚠️ Saisis une requête pour rechercher !")

# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("*Projet NLP — DistilBERT + TF-IDF fine-tunés sur 24k avis assurances*")
