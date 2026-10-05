# Reproducibilidad

## Experimento

- Dataset oficial PolyAI, revisión `57ec275d8078af65b7731c2a98be812d844a6d6b`.
- 10.003 consultas de entrenamiento, 3.080 de prueba, 77 etiquetas.
- Validación: primera partición de StratifiedGroupKFold de cinco folds; grupos por texto normalizado, semilla 42.
- Clasificador: `distilbert/distilroberta-base`, ajuste de todos los pesos y cabeza de 77 clases.
- Tasas evaluadas: 2e-5, 3e-5 y 5e-5; cinco épocas máximas. Selección por F1 macro de validación; ganador 5e-5.
- Falcon: `tiiuae/falcon-7b-instruct`, inferencia NF4/FP16, sin entrenamiento.

El notebook conserva configuración, versiones, salidas y referencias. Los datos se descargan desde las fuentes fijadas; no se redistribuye una copia del dataset en este repositorio. Los resultados guardados no son un entrenamiento recién ejecutado por CI.

## Pesos

Se necesita el directorio `best/` generado por el notebook. Contiene configuración, tokenizer y `model.safetensors`. El importador verifica los nombres de las 77 etiquetas y rechaza rutas inseguras de ZIP; no carga `training_args.bin` (pickle).

El checkpoint de la demo tiene 82.177.613 parámetros. SHA256 de `model.safetensors`:

```text
30f97de93f9a423cf72b5749a5f05b81dca3a976f92857e4cc66a7f3874bd6d8
```

Veinte predicciones archivadas comparadas con el checkpoint local conservaron sus etiquetas. La comparación no constituye una reevaluación del test completo. Tres consultas nuevas se probaron adicionalmente en CPU local y Modal.

## Revisión de Falcon

Las anotaciones académicas son asistidas y aún requieren una revisión humana independiente si se pretende presentarlas como tal. El prompt de la web es una variante delimitada por JSON (`quoted-query-v1`), no el prompt ganador del notebook y no hereda sus métricas. Falcon está deshabilitado en el despliegue actual; habilitarlo requerirá una nueva prueba y revisión.
