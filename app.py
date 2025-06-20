import streamlit as st
import ezc3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
import tempfile

st.set_page_config(page_title="Analyse CRP", layout="centered")
st.title("🧠 Analyse CRP - Interface interactive")

# 1. Upload des fichiers .c3d
st.header("1. Importer un ou plusieurs fichiers .c3d")
uploaded_files = st.file_uploader("Choisissez un ou plusieurs fichiers .c3d", type="c3d", accept_multiple_files=True)

if uploaded_files:
    selected_file = st.selectbox("Choisissez un fichier pour l'analyse", uploaded_files, format_func=lambda x: x.name)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(selected_file.read())
        tmp_path = tmp.name

    c3d = ezc3d.c3d(tmp_path)
    labels = c3d['parameters']['POINT']['LABELS']['value']
    freq = c3d['header']['points']['frame_rate']
    first_frame = c3d['header']['points']['first_frame']
    points = c3d['data']['points']
    n_frames = points.shape[2]
    time = np.arange(n_frames) / freq + first_frame / freq

    st.success("Fichier .c3d chargé avec succès !")

    # 2. Sélection des marqueurs
    st.header("2. Sélection des marqueurs")
    st.write("Labels disponibles :", labels)

    heel_marker = st.selectbox("Marqueur du talon (pour détection des cycles)", labels)
    marker1 = st.selectbox("Marqueur d’intérêt 1 (ex. hanche)", labels)
    marker2 = st.selectbox("Marqueur d’intérêt 2 (ex. épaule)", labels)

    def extract_and_normalize_cycles(points, labels, marker_name, valid_cycles):
        idx = labels.index(marker_name)
        signal = points[0, idx, :]  # Plan sagittal
        all_cycles = []

        for i, (start, end) in enumerate(valid_cycles):
            segment = signal[start:end]
            x_original = np.linspace(0, 100, num=len(segment))
            x_interp = np.linspace(0, 100, num=100)

            try:
                f = interp1d(x_original, segment, kind='cubic')
                normalized = f(x_interp)
                if not np.isnan(normalized).any():
                    all_cycles.append(normalized)
                else:
                    st.warning(f"⚠️ Cycle {i+1} pour {marker_name} contient des NaN → exclu automatiquement.")
            except Exception as e:
                st.warning(f"⚠️ Erreur d'interpolation cycle {i+1} pour {marker_name} : {e}")

        return np.array(all_cycles)

    if st.button("Lancer la détection + extraction pour les 2 marqueurs"):
        try:
            # --- Détection des cycles ---
            idx_heel = labels.index(heel_marker)
            z_heel = points[2, idx_heel, :]
            inverted_z = -z_heel
            min_distance = int(freq * 0.8)
            peaks, _ = find_peaks(inverted_z, distance=min_distance, prominence=1)

            cycle_starts = peaks[:-1]
            cycle_ends = peaks[1:]
            min_duration = int(0.5 * freq)
            valid_cycles = [(start, end) for start, end in zip(cycle_starts, cycle_ends) if (end - start) >= min_duration]
            n_cycles = len(valid_cycles)

            fig1, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(time, z_heel, label=f"Z ({heel_marker})")
            ax1.plot(time[peaks], z_heel[peaks], "ro", label="Début de cycle")
            ax1.set_title(f"Détection des cycles via {heel_marker}")
            ax1.set_xlabel("Temps (s)")
            ax1.set_ylabel("Hauteur (Z)")
            ax1.grid(alpha=0.3)
            ax1.legend()
            st.pyplot(fig1)
            st.success(f"{n_cycles} cycles valides détectés.")

            st.header("3. Analyses")

            marker1_cycles = extract_and_normalize_cycles(points, labels, marker1, valid_cycles)
            marker2_cycles = extract_and_normalize_cycles(points, labels, marker2, valid_cycles)

            if marker1_cycles.size > 0:
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                x = np.linspace(0, 100, 100)
                for cycle in marker1_cycles:
                    ax2.plot(x, cycle, alpha=0.5)
                mean1 = np.mean(marker1_cycles, axis=0)
                ax2.plot(x, mean1, color="black", linewidth=2.5, label="Moyenne")
                ax2.set_title(f"{marker1} - Cycles normalisés (plan sagittal)")
                ax2.set_xlabel("Cycle (%)")
                ax2.set_ylabel("Angle (°)")
                ax2.grid(alpha=0.3)
                ax2.legend()
                st.pyplot(fig2)
            else:
                st.warning(f"Aucun cycle valide pour {marker1} après nettoyage.")

            if marker2_cycles.size > 0:
                fig3, ax3 = plt.subplots(figsize=(10, 5))
                x = np.linspace(0, 100, 100)
                for cycle in marker2_cycles:
                    ax3.plot(x, cycle, alpha=0.5)
                mean2 = np.mean(marker2_cycles, axis=0)
                ax3.plot(x, mean2, color="black", linewidth=2.5, label="Moyenne")
                ax3.set_title(f"{marker2} - Cycles normalisés (plan sagittal)")
                ax3.set_xlabel("Cycle (%)")
                ax3.set_ylabel("Angle (°)")
                ax3.grid(alpha=0.3)
                ax3.legend()
                st.pyplot(fig3)
            else:
                st.warning(f"Aucun cycle valide pour {marker2} après nettoyage.")

            # --- VITESSE ANGULAIRE ---
            if marker1_cycles.size > 0:
                marker1_velocity = np.gradient(marker1_cycles, axis=1)
                mean_marker1_velocity = np.mean(marker1_velocity, axis=0)

                fig4, ax4 = plt.subplots(figsize=(10, 5))
                x = np.linspace(0, 100, 100)
                ax4.plot(x, mean_marker1_velocity, color="purple", linewidth=2.5, label=f"Vitesse angulaire moyenne ({marker1})")
                ax4.set_title(f"Vitesse angulaire - {marker1} (plan sagittal)")
                ax4.set_xlabel("Cycle de marche (%)")
                ax4.set_ylabel("Vitesse angulaire (°/s)")
                ax4.grid(alpha=0.3)
                ax4.legend()
                st.pyplot(fig4)

                # Plan de phase
                scaler = MinMaxScaler(feature_range=(-1, 1))
                marker1_angle_norm = scaler.fit_transform(mean1.reshape(-1, 1)).flatten()
                marker1_velocity_norm = scaler.fit_transform(mean_marker1_velocity.reshape(-1, 1)).flatten()

                fig_phase1, ax_phase1 = plt.subplots(figsize=(6, 6))
                ax_phase1.plot(marker1_angle_norm, marker1_velocity_norm, color="blue")
                ax_phase1.axhline(0, color="black", linewidth=0.8)
                ax_phase1.axvline(0, color="black", linewidth=0.8)
                ax_phase1.set_title(f"Angle polaire de {marker1}")
                ax_phase1.set_xlabel(f"Angle normalisé de {marker1}")
                ax_phase1.set_ylabel(f"Vitesse angulaire normalisée de {marker1}")
                ax_phase1.grid(True)
                ax_phase1.axis("equal")
                st.pyplot(fig_phase1)

                # Angle de phase
                marker1_phase_rad = np.arctan2(marker1_velocity_norm, marker1_angle_norm)
                marker1_phase_unwrapped = np.unwrap(marker1_phase_rad)
                marker1_phase_deg = np.degrees(marker1_phase_unwrapped)

                fig_angle1, ax_angle1 = plt.subplots(figsize=(10, 4))
                ax_angle1.plot(np.linspace(0, 100, 100), marker1_phase_deg, color="green")
                ax_angle1.set_title(f"Angle de phase (déplié) - {marker1}")
                ax_angle1.set_xlabel("Cycle de marche (%)")
                ax_angle1.set_ylabel("Angle de phase (°)")
                ax_angle1.grid(True)
                st.pyplot(fig_angle1)

            if marker2_cycles.size > 0:
                marker2_velocity = np.gradient(marker2_cycles, axis=1)
                mean_marker2_velocity = np.mean(marker2_velocity, axis=0)

                fig5, ax5 = plt.subplots(figsize=(10, 5))
                x = np.linspace(0, 100, 100)
                ax5.plot(x, mean_marker2_velocity, color="orange", linewidth=2.5, label=f"Vitesse angulaire moyenne ({marker2})")
                ax5.set_title(f"Vitesse angulaire - {marker2} (plan sagittal)")
                ax5.set_xlabel("Cycle de marche (%)")
                ax5.set_ylabel("Vitesse angulaire (°/s)")
                ax5.grid(alpha=0.3)
                ax5.legend()
                st.pyplot(fig5)

                scaler = MinMaxScaler(feature_range=(-1, 1))
                marker2_angle_norm = scaler.fit_transform(mean2.reshape(-1, 1)).flatten()
                marker2_velocity_norm = scaler.fit_transform(mean_marker2_velocity.reshape(-1, 1)).flatten()

                fig_phase2, ax_phase2 = plt.subplots(figsize=(6, 6))
                ax_phase2.plot(marker2_angle_norm, marker2_velocity_norm, color="blue")
                ax_phase2.axhline(0, color="black", linewidth=0.8)
                ax_phase2.axvline(0, color="black", linewidth=0.8)
                ax_phase2.set_title(f"Angle polaire de {marker2}")
                ax_phase2.set_xlabel(f"Angle normalisé de {marker2}")
                ax_phase2.set_ylabel(f"Vitesse angulaire normalisée de {marker2}")
                ax_phase2.grid(True)
                ax_phase2.axis("equal")
                st.pyplot(fig_phase2)

                marker2_phase_rad = np.arctan2(marker2_velocity_norm, marker2_angle_norm)
                marker2_phase_unwrapped = np.unwrap(marker2_phase_rad)
                marker2_phase_deg = np.degrees(marker2_phase_unwrapped)

                fig_angle2, ax_angle2 = plt.subplots(figsize=(10, 4))
                ax_angle2.plot(np.linspace(0, 100, 100), marker2_phase_deg, color="green")
                ax_angle2.set_title(f"Angle de phase (déplié) - {marker2}")
                ax_angle2.set_xlabel("Cycle de marche (%)")
                ax_angle2.set_ylabel("Angle de phase (°)")
                ax_angle2.grid(True)
                st.pyplot(fig_angle2)

            # --- CALCUL DU CRP ---
            if marker1_cycles.size > 0 and marker2_cycles.size > 0:
                crp_rad = marker1_phase_unwrapped - marker2_phase_unwrapped
                crp_norm = crp_rad / np.pi

                fig_crp, ax_crp = plt.subplots(figsize=(10, 5))
                ax_crp.plot(np.linspace(0, 100, 100), crp_norm, color='brown', label='CRP normalisé [-1,1]')
                ax_crp.axhline(0, color='black', linestyle='--', linewidth=0.8)
                ax_crp.set_title(f"Continuous Relative Phase (CRP) normalisé entre {marker1} et {marker2}")
                ax_crp.set_xlabel("Cycle de marche (%)")
                ax_crp.set_ylabel("CRP (normalisé)")
                ax_crp.set_ylim(-3, 3)
                ax_crp.grid(True)
                ax_crp.legend()
                st.pyplot(fig_crp)

                # Statistiques CRP
                crp_mean_val = np.mean(crp_norm)
                crp_std = np.std(crp_norm)
                crp_min = np.min(crp_norm)
                crp_max = np.max(crp_norm)
                crp_min_pos = np.argmin(crp_norm)
                crp_max_pos = np.argmax(crp_norm)

                st.markdown("### 📊 Statistiques du CRP")
                st.write(f"**CRP moyen** : {crp_mean_val:.3f}")
                st.write(f"**Écart-type** : {crp_std:.3f}")
                st.write(f"**CRP minimum** : {crp_min:.3f} à {crp_min_pos} % du cycle")
                st.write(f"**CRP maximum** : {crp_max:.3f} à {crp_max_pos} % du cycle")
--- INTERPRÉTATION ---
                with st.expander("**🧠 Interprétation des résultats CRP**", expanded = False):
                    st.markdown("""
                Le **CRP (Continuous Relative Phase)** permet d’évaluer la **coordination dynamique** entre deux segments corporels (par exemple, une hanche et une épaule) pendant le cycle de marche. Il est calculé à partir de la **différence entre leurs angles de phase**.

                #### **Comment interpréter les valeurs du CRP ?**
                - **Valeurs proches de 0** → les deux segments sont **en phase** : ils bougent **de manière synchronisée**.
                - **Valeurs proches de ±1** → les segments sont **en opposition de phase** : ils bougent **en décalage** (l’un est en avance ou en retard par rapport à l’autre).
                - **Valeurs positives CRP > 0** : le **premier segment/marqueur** (sélectionné en premier dans l’appli) est **en avance**.
                - **Valeurs négatives CRP < 0** : le **premier segment/marqueur** est **en retard**.
                            
                #### **Forme de la courbe CRP :**
                - Une **oscillation régulière** → coordination **stable et cyclique**.
                - Une courbe **très variable, plate ou inconstante** → coordination **instable, adaptative ou perturbée/atypique**.

                #### ⚠️ **Attention à l’ordre des marqueurs !**
                Le CRP dépend de l’ordre dans lequel les marqueurs sont sélectionnés :
                - Si on compare la **hanche (1er)** à l’**épaule (2ème)**, un CRP positif = hanche en avance.
                - Inverser l’ordre inverse également le **signe du CRP**.

                Toujours interpréter les résultats **en gardant cela en tête** !
                """)

        except Exception as e:
            st.error(f"Erreur pendant l'analyse : {e}")
