# F4 mínima: gesto desde texto

Claude, 18/11/2026. Todo local y gratis: numpy, Jolt (`jolt-physics` 1.1.1, MIT, el mismo de F3) y Blender 5.2.3
(solo para la vista previa, en el PC viejo). No se añadió ninguna dependencia.

> Ojo con el nombre: en el PLAN, «F4» era la fuente de dos teléfonos con MediaPipe. Esto es otra cosa, una F4
> **mínima** que pidió Daniel el 18/11: describir el resultado con una frase y obtener el clip.

## En corto
- **La orden:**
  ```bash
  python -m forja.gesto "pick up box from ground" --personaje MODELO.glb --salida CARPETA
  ```
  Escribe en `CARPETA` el GLB del personaje con un clip esquelético nuevo y la caja animada como nodo aparte, además de
  las anotaciones y el informe de medidas.
- **Medido: 22–29 s de reloj de la frase al GLB, en la CPU de la casita** (8 GB, con otras sesiones abiertas; `time`
  sobre el proceso entero, con el arranque de Python incluido). La meta era menos de 60 s.
- **Resultado del ejemplo** (`salidas/gesto/`, un humanoide del juego de 18 huesos con la ropa pesada a la piel):
  - **pies:** deslizamiento máximo de 0,10 cm por cuadro y 0,5 cm acumulados mientras están apoyados;
  - **suelo:** la malla deformada baja como mucho 0,18 cm (en el talón, al aterrizar un paso);
  - **manos sobre la caja:** mientras la sujetan, la palma queda a +0,2 mm de su cara como mucho, y entra 0,3 mm como mucho;
  - **la caja:** no la atraviesa ninguna otra parte del cuerpo (0,0 cm con la malla);
  - **la caja contra el suelo:** 0,006 cm.
- **Vista previa:** `docs/video/f4_gesto_recoger_caja.mp4`, 1920×1080, 12,3 s. Son 7 s en tercera persona con la cámara
  quieta y 5,3 s en primera persona.

## 1. Qué hace, paso a paso

| Paso | Archivo | Qué hace |
|---|---|---|
| Frase | `forja/gesto/frase.py` | Plantillas de verbo + objeto con expresiones regulares, en inglés y en español: *pick up / lift / grab (the) box (from ground)*, *put / set / place box down*, *recoge / levanta la caja*, *deja / baja la caja*. Normaliza (minúsculas, sin artículos) y, si no encaja nada, lo dice y no adivina |
| Personaje | `forja/gesto/humanoide.py` | Lee el GLB sin Blender: el esqueleto pasa al `Rig` de Forja (Z arriba, pose del GLB = θ 0), la piel lineal y la malla. Deduce hacia dónde mira y el sentido de las bisagras (rodilla, codo) de la propia geometría |
| Metas | `forja/gesto/gestos.py` | El gesto como **resultados**: dónde están la caja y los pies, a qué altura se agarra la caja y cuándo pasa cada cosa. Las poses clave del cuerpo (altura y retroceso de la pelvis, inclinación del tronco) **se buscan**, no se escriben a mano (§2) |
| Cuerpo | `forja/gesto/cuerpo.py` | La IK de Forja (`ajuste/ik.py`, CadenaIK con límites articulares) en piernas y brazos. Tobillo y muñeca son metas fuertes; rodilla y codo, débiles (solo eligen el plano). Pie y mano se orientan de forma analítica: el pie plano, la palma hacia dentro y los dedos hacia abajo. La inclinación del tronco se reparte así: 55 % cadera, 25 % `spine` y 20 % `chest` |
| Física | `forja/gesto/fisica_caja.mjs` | Jolt. La caja (6 kg) es un cuerpo dinámico con gravedad que se asienta en el suelo. En el cuadro del agarre, una restricción fija la une a un cuerpo cinemático que sigue el punto medio de las dos palmas, así que desde ahí la caja sigue a las manos como cuerpo físico. Las partes del cuerpo, como cascos convexos, se consultan contra la caja y el suelo |
| Medida y limpieza | `forja/gesto/__main__.py` | Con la **malla deformada**: suelo, hueco o penetración de cada palma con su cara, cuerpo dentro de la caja y deslizamiento de los pies (§3) |
| GLB | `forja/gesto/__main__.py` (`exportar`) | Una copia del GLB del personaje con su malla, su piel, sus texturas y su rig, y **un solo clip** (`recoger_caja`, 30 fps) sobre los 17 huesos del rig, además del nodo `caja` animado. Los nombres de nodo que llevaban el del personaje pasan a `personaje_*` |
| Vista previa | `forja/gesto/camaras.py`, `blender_video.py`, `video.py`, `scripts/pc-viejo/f4g_video.sh` | Importa el GLB en Blender como lo haría un motor y renderiza con Eevee: la toma en tercera persona y la de primera, con la cámara a la altura de los ojos y 3 cm por delante de la cara |
| Telemetría | `forja/gesto/telemetria.py` | Banderas por segundo para el crítico de video: `desliza`, `bajo-suelo`, `atraviesa`, `mano-al-aire`, `caja-al-aire` |

**El clip «recoger caja»** (7 s):
1. Quieto (0–0,4 s).
2. Dos pasos para acercarse 30 cm: uno con la izquierda y otro con la derecha. Cada pie se abre 3 cm y el peso pasa al pie
   de apoyo (0,4–1,6 s).
3. Se agacha y las manos van a 6 cm por fuera de las caras de la caja (1,9–3,05 s).
4. Cierra las manos sobre la caja (hasta 3,35 s) y sujeta (3,35–3,6 s).
5. Levanta (3,6–5,25 s): cerca del suelo, la caja sube en vertical, lejos de las rodillas; arriba se acerca al cuerpo con un
   leve arco hacia delante.
6. Queda de pie con la caja delante, respirando (hasta 7 s). La cabeza mira la caja y, al final, al frente.

**«put box down»** también funciona (6,5 s): baja la caja hasta 4 mm del suelo, la suelta (la física la deja caer y
asentarse), abre las manos y se levanta. Está medido igual (palmas a ≤ 0,1 mm y −0,12 mm, suelo −0,22 cm, cuerpo en la
caja 0,0 cm), pero **no tiene video**.

## 2. Las poses clave se buscan
Para agarrar y para cargar se prueba una rejilla de candidatos: altura de la pelvis, cuánto va la cadera hacia atrás e
inclinación. Primero va una rejilla gruesa y después una fina de 3×3×3 alrededor del mejor. Todos los candidatos se
resuelven **en lote** con la misma IK, y gana el de menor coste:

`coste = 10⁵·error² (muñecas y tobillos, m) + inclinación² + 20·(inclinación − 48°)₊² + 0,5·rodilla² + 300·(5 cm − margen del centro de masas sobre los pies)₊²`

El centro de masas usa las fracciones de Winter por segmento y, al cargar, suma la caja.

En el ejemplo:
- **agarre:** pelvis a 0,36 m, cadera 12 cm atrás, tronco a 50° y rodillas a 137°. Es una sentadilla profunda; el centro de
  masas queda 6,5 cm dentro del apoyo;
- **carga:** de pie, sin inclinación y con las rodillas a 13°; el centro de masas con la caja queda 8 cm dentro del apoyo.

## 3. La limpieza, con números (`salidas/gesto/informe.json`)
La limpieza corrige **el plan** (dónde está la caja) y **las metas** (muñecas, tobillos), y vuelve a resolver:

| Vuelta | Qué encontró | Qué cambió |
|---|---|---|
| 1 | Con la caja a 8 cm de las puntas, la tibia derecha entraba **6,1 cm** en la caja al empezar a levantar (cuadro 119) | La caja se aleja: 8 → 15,6 cm |
| 2 | Con la caja en brazos, el vientre entraba **3,4 cm** en ella (cuadro 158) | La caja en brazos va más adelante: 30 → 34,9 cm |
| 3 | Nada dentro de la caja | — |

En cada vuelta, además:
- **Suelo con la malla:** si la piel de tibia y pie baja del suelo, ese tobillo sube lo que falta (−0,32 → −0,18 cm).
- **Palmas:** el hueco o la penetración medidos con la malla se quitan de la meta de la muñeca, suavizados en el tiempo. En
  este ejemplo no hizo falta: el punto de contacto de la palma, sacado de la malla de la mano, ya cae en la cara a menos de
  0,3 mm.

**Jolt y la malla no dicen lo mismo.** Los cascos convexos de los muslos «tocan» la caja hasta 2,4 cm en 16 cuadros y
los de los pies «tocan» el suelo hasta 1 cm. Es lo que añade el casco convexo, porque la malla real queda fuera (0,0 cm y
−0,18 cm). Se informa de las dos medidas y la que manda es la de la malla.

**Reloj por paso** (la vuelta más larga): metas 0,6 s, IK 1,4 s, Jolt 0,4 s, medida con la malla y corrección del suelo
5–6 s. El GLB se escribe en 0,03 s. La vuelta 1 cuesta lo mismo aunque luego se repita: si la caja empezara donde acaba
la limpieza, saldría en ~8 s.

## 4. La vista previa y el crítico
`docs/video/f4_gesto_recoger_caja.mp4`: Blender importa el GLB de la salida (las acciones `recoger_caja` de la armadura
y de la caja), con suelo a cuadros, sol y Eevee a 32 muestras, 1920×1080 y 30 fps. El render tarda ~3 min en la 3070;
Cycles a 16 muestras iba a 8 s por cuadro (50 min) y se cambió.

**Crítico** (`docs/datos/f4/critica_gesto.md`, imágenes en `docs/img/f4_gesto_hoja.jpg` y `f4_gesto_cerca.jpg`):
- **Primera pasada:**
  - una grave: la cámara subjetiva caía dentro de la cabeza (la malla llega a 16 cm del hueso y la cámara iba a 11);
  - una media: con la cámara pegada a la cabeza, la caja salía del cuadro 2 s al final.
- **Arreglo:** la cámara va 3 cm por delante de la cara y los ojos siguen la caja.
- **Segunda pasada:** sin graves y sin banderas en `segundos.md` ni en la telemetría.
  - Queda una media: sin dedos, las palmas sostienen la caja abiertas.
  - Quedan leves: la caja sube un instante antes que las rodillas, el personaje ocupa 1/3 del alto y los faldones de la
    bata se pliegan en la sentadilla.
- **Revisado a mano a resolución completa:**
  - los pies no se deslizan;
  - las palmas tocan las caras de la caja;
  - nada atraviesa el suelo.

## 5. Cómo repetirlo
```bash
cd ~/Projects/forja
M=…/frontend/src/assets/models/creatures/<humanoide>.glb     # solo lectura
PYTHONPATH=. uv run --no-project --with numpy python -m forja.gesto "pick up box from ground" --personaje $M --salida salidas/gesto
PYTHONPATH=. uv run --no-project --with numpy python -m forja.gesto.camaras salidas/gesto salidas/gesto/camaras.json --primera 50:210
PYTHONPATH=. uv run --no-project --with numpy python -m forja.gesto.telemetria $M salidas/gesto salidas/gesto/telemetria-segundos.json --tomas "0:210@0,50:210@7"
# vista previa en el PC viejo (con candado; borrar la tarea al terminar)
scripts/local/pc.sh sync && scripts/local/pc.sh sync-gesto salidas/gesto
scripts/local/pc.sh run f4g_video 'GLB=recoger_caja.glb' 'FRASE="pick up box from ground"'
scripts/local/pc.sh fin f4g_video
scp …:%USERPROFILE%/forja/f4g/vista_previa.mp4 . && ffmpeg -i vista_previa.mp4 -c:v libx264 -crf 20 -pix_fmt yuv420p docs/video/f4_gesto_recoger_caja.mp4
python3 -I <critico-de-video>/medir.py docs/video/f4_gesto_recoger_caja.mp4 CRITICA/ --fps 4 --telemetria salidas/gesto/telemetria-segundos.json
```
`npm install` en la raíz instala `jolt-physics`. Las dos órdenes de `uv` no instalan nada en el sistema.

## 6. Lo que NO hace todavía, en claro
- **No entiende lenguaje libre.** Son plantillas: hoy, *recoger caja* y *dejar caja* con sus sinónimos. «push box», «run» o
  «shoot» no existen, y la orden dice que no los entiende. Añadir un verbo es escribir su plan de metas en `gestos.py`, no
  entrenar nada.
- **La caja es siempre la misma:** 36 × 26 × 26 cm y 6 kg, en el suelo delante del personaje y en línea recta, sin giro.
  La frase no puede decir dónde está la caja ni cuánto pesa, y el peso no cambia la postura (no hay dinámica del cuerpo).
- **El cuerpo no es una simulación física.** Jolt mueve la caja y detecta choques; el cuerpo sale de metas + IK + una
  comprobación de equilibrio estático (centro de masas sobre los pies). No hay ragdoll ni fuerzas en las articulaciones.
- **Sin dedos.** El rig del juego acaba en la mano: la palma se apoya plana en la cara de la caja y los dedos no se
  cierran. De cerca se nota que la sujeta «con las palmas», no con un agarre.
- **Los pasos son un patrón fijo** (dos pasos cortos de 30 cm, siempre hacia delante). Para llegar a una caja lejana o de
  lado hace falta la marcha procedural humanoide, que no está.
- **Un personaje probado.** Funciona con esqueletos humanoides que tengan `pelvis/spine/chest/neck/head`,
  `thigh/shin/foot` y `upper_arm/forearm/hand` en `_L/_R` (los del juego), mirando a +Z de glTF. Con otros nombres falla
  con un mensaje.
- **La primera persona** es una cámara en la cabeza del personaje: no es una cámara de juego en primera persona.
- **El cuerpo en cuclillas profundas** (pelvis a 0,36 m) sale de la búsqueda y no está calibrado contra movimiento real.
  Una persona real se acerca más a la caja y se agacha algo menos: el castigo de la inclinación y la sentadilla se ajustó
  a ojo.
- **La ropa** (la bata) se deforma con la piel lineal del modelo. En la sentadilla, los faldones se pliegan mal; eso es del
  pintado de pesos del GLB, no del clip.
