# Crítica: `docs/video/f4_gesto_recoger_caja.mp4` (18/11/2026)

Hecha con la skill critico-de-video (`medir.py --fps 4` y la telemetría `telemetria-segundos.json` de
`forja/gesto/telemetria.py`). `medir.py` reduce a vertical (270×480), así que también se miraron hojas apaisadas a
4 cuadros por segundo y recortes a resolución completa: `docs/img/f4_gesto_hoja.jpg` y `docs/img/f4_gesto_cerca.jpg`.

## Primera pasada (video v1)

| t | qué se ve | plano | reglas | gravedad | arreglo |
|---|---|---|---|---|---|
| 7,0–12,3 | toma subjetiva: la cámara cae dentro de la cabeza; solo se ven la cara por dentro y las gafas | primera persona | R4 R14 | grave | la cámara se mueve 3 cm por delante de la cara (la malla de la cabeza llega a 16 cm del hueso; la cámara iba a 11 cm) |
| (v1 corregido) 11–12,3 | con la cámara siguiendo la cabeza, la caja sale del cuadro al final: 2 s de suelo vacío | primera persona | R11 | media | los ojos siguen la caja: la mirada mezcla 60 % la caja y 40 % la cabeza |

## Segunda pasada (video final)

`segundos.md`: sin banderas (ni oscuro, ni borroso, ni suelo sin textura, ni cortes seguidos). Telemetría: sin `desliza`,
`bajo-suelo`, `atraviesa`, `mano-al-aire` ni `caja-al-aire` en ningún segundo.

| t | qué se ve | plano | reglas | gravedad | arreglo |
|---|---|---|---|---|---|
| 3,0–5,2 | de cerca, las palmas apoyan planas en las caras de la caja con los dedos abiertos: la sostiene a presión, sin cerrar la mano | tercera | R2 | media | el rig del juego no tiene dedos; hace falta un rig con falanges o una pose de mano cerrada pintada en el modelo |
| 4,3–4,8 | la caja sube hasta el pecho antes de que las rodillas se estiren: un instante de cuclillas con la caja en alto | tercera | R6 | leve | la caja sube a la vez que la cadera (hoy el trayecto se adelanta) |
| 0–7 | el personaje ocupa ~1/3 del alto: se lee, pero los pies y las manos se ven pequeños | tercera | R8 | leve | un segundo plano de detalle (pies y manos) o una cámara más cerrada; se quiso cámara quieta y todo el recorrido en cuadro |
| 2,5–4 | los faldones de la bata se pliegan en ángulos duros en la sentadilla | tercera | R10 | leve | es el pintado de pesos del GLB, no el clip; repintar en el modelo con el sí de Daniel |
| 7,0 | el paso de tercera a primera persona es un corte seco, sin cartela | — | R11 | leve | una cartela de 0,5 s o una cortinilla |

**Resumen en tres líneas**
- Lo peor: las manos sin dedos sostienen la caja con las palmas abiertas, y de cerca se nota.
- Lo mejor: los pies quietos en sus apoyos, nada atraviesa el suelo ni la caja, la caja sube con las manos sin
  teletransportarse y la subjetiva muestra las dos manos en la caja.
- Se puede enseñar: sin graves. Las medias quedan dichas en el documento.

**Lo que no se pudo juzgar desde las hojas:** el deslizamiento fino de los pies (≤ 1 mm por cuadro, según la medida con la
malla, no es visible) y si la caja tiembla al subir. La caja que devuelve Jolt va unida a las manos con una restricción
fija, y en la hoja a 4 cuadros por segundo no se ve temblor.

**Regla propuesta:** ninguna nueva. Se confirma la de F3 para `medir.py`: hace falta un modo apaisado (16:9), porque las
hojas salen estrechas.
