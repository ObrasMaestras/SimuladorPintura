# Botón grande para pintar
            if st.button("🎨 PINTAR ESTA PARED", type="primary", use_container_width=True):
                with st.spinner("🤖 Detectando la pared con inteligencia artificial..."):
                    predictor = cargar_predictor()
                    
                    if predictor is not None:
                        try:
                            imagen_np = np.array(imagen)
                            predictor.set_image(imagen_np)
                            punto_input = np.array([[x, y]])
                            etiqueta_input = np.array([1])
                            
                            mascaras, scores, _ = predictor.predict(
                                point_coords=punto_input,
                                point_labels=etiqueta_input,
                                multimask_output=True,
                            )
                            
                            mejor_idx = np.argmax(scores)
                            mascara = mascaras[mejor_idx]
                            
                            st.session_state.areas_pintadas.append({
                                'mascara': mascara,
                                'color': color_actual,
                                'coordenadas': (x, y)
                            })
                            
                            st.success("🎉 ¡Pared pintada!")
                            st.balloons()
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                    else:
                        st.error("❌ No se pudo cargar el modelo de IA")
        
        
