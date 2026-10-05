# Modal

## Estado comprobado

Endpoint: https://enriquemunozdev--banking77-enrique-munoz-web.modal.run/

La demo utiliza el checkpoint ajustado en CPU. La cuenta de prueba observada tenía USD 1 de crédito y límite bruto USD 1; tras las pruebas el panel mostraba USD 0,01 de créditos consumidos y USD 0 en cargos (valores redondeados, no estado de facturación en tiempo real). No se añadió tarjeta ni se elevó el presupuesto. Modal rechazó explícitamente GPU T4 sin método de pago.

## Publicar en tu propia cuenta

1. Instala las dependencias de `web/backend/requirements.txt` y autentica la CLI oficial con `modal token new`. No añadas credenciales al código.
2. Verifica en el panel los créditos, el límite bruto de uso y el límite neto de gasto. No asumir USD 30 desbloqueados ni coste cero por publicidad del plan.
3. Crea el Secret `banking77-demo-config` con `B77_DEMO_ACCESS_SHA256` (SHA256 de un código privado aleatorio) y `B77_ALLOWED_ORIGINS`. Para frontend y API del mismo origen, este último puede estar vacío. No publiques el código ni el hash en el frontend.
4. Desde `web/`, define `BANKING77_MODEL_DIR` apuntando a `best/` y ejecuta:

```sh
export BANKING77_MODEL_DIR=/ruta/privada/banking77-best
export B77_DEPLOY_BUDGET_CONFIRMED=yes
export B77_ENABLE_FALCON=no
modal deploy backend/modal_app.py
```

El indicador de aprobación no configura facturación: solo confirma que la revisaste. El endpoint web público exige código en los POST de inferencia; la salud y los archivos estáticos son públicos. No usa proxy authentication de Modal porque el control está en FastAPI.

## Consumo y seguridad

- CPU: 2 núcleos, RAM 4 GiB; mínimo cero, máximo un contenedor; apagado tras 30 segundos sin actividad.
- Máximo 1.200 caracteres, una inferencia a la vez, 12 clasificaciones/minuto y 80/hora por contenedor.
- No se sirven archivos de backend, pesos ni pruebas.
- El código se mantiene solamente en memoria de la página; se transmite por HTTPS y el backend compara hashes en tiempo constante.
- La aplicación no guarda consultas en base de datos. No se garantiza ausencia de registros operativos del proveedor.
- Cuando se agote el crédito, la disponibilidad puede detenerse. No es alojamiento gratuito ilimitado.

## Falcon opcional

No habilitado en la demo actual. Solo si tu cuenta permite GPU y has verificado el presupuesto:

```sh
export B77_ENABLE_FALCON=yes
modal run backend/modal_app.py::prefetch_falcon
modal deploy backend/modal_app.py
```

La precarga descarga Falcon en CPU a un volumen persistente. La inferencia usa T4, NF4/FP16 y se apaga tras 30 segundos. Carga, arranque y tiempo caliente pueden consumir crédito. Requiere validar generación real, parámetros y respuestas antes de declarar operativo el LLM. No se ha hecho esta validación cloud en la cuenta actual.
