import streamlit as st
from PIL import Image
import pytesseract
import pandas as pd
import re

st.set_page_config(page_title="Estrai Referto", layout="centered")
st.title("🧪 Estrazione automatica di valori da referti ematici")

st.markdown("Carica o fotografa un referto: l'app estrarrà **Test**, **Valore** e **Unità di misura**.")

uploaded_file = st.file_uploader("📷 Carica immagine (JPG/PNG)", type=["jpg", "jpeg", "png"])

def estrai_valori(testo):
    righe = testo.split("\n")
    risultati = []
    pattern = r"([A-Za-z0-9 #\(\)/%µ\^\-]+?)\s+([\d.,]+)\s*([a-zA-Z/µ^%³]+)?"
    
    for riga in righe:
        match = re.match(pattern, riga.strip())
        if match:
            test, valore, unita = match.groups()
            try:
                valore_float = float(valore.replace(",", "."))
                risultati.append({
                    "Test": test.strip(),
                    "Valore": valore_float,
                    "Unità": unita or ""
                })
            except:
                continue
    return risultati

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Referto caricato", use_column_width=True)

    with st.spinner("📖 Estrazione in corso..."):
        testo = pytesseract.image_to_string(image, lang="eng+ita")
        dati = estrai_valori(testo)

    if dati:
        df = pd.DataFrame(dati)
        st.success("✅ Valori estratti con successo")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Scarica come CSV", data=csv, file_name="valori_referto.csv", mime="text/csv")
    else:
        st.warning("Nessun valore riconosciuto.")