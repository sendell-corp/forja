// Forja · gesto/fisica_caja.mjs — la caja como cuerpo físico y las colisiones del cuerpo con Jolt (jolt-physics 1.1.1, MIT).
//
//   node forja/gesto/fisica_caja.mjs ENTRADA.json SALIDA.json
//
// ENTRADA: { "caja": {"medidas": [ancho, fondo, alto], "masa": kg, "pose0": [x,y,z, qx,qy,qz,qw]},
//            "fps": 30, "sub": 8,
//            "agarre": [cuadro_desde, cuadro_hasta|null]    tramo en que las manos sujetan la caja
//            "mano": [[x,y,z, qx,qy,qz,qw], …]              pose del agarre (entre las dos palmas) por cuadro
//            "cascos": {hueso: [[x,y,z], …]}                puntos de la malla de cada hueso en reposo (Forja, Z arriba)
//            "cuadros": [{hueso: [r00..r22, tx,ty,tz]}, …]  reposo → pose de cada hueso por cuadro }
// SALIDA:  { "caja": [[x,y,z, qx,qy,qz,qw], …] por cuadro,
//            "caja_cuerpo": [[cuadro, hueso, profundidad_m], …]   partes del cuerpo dentro de la caja
//            "suelo": [[cuadro, hueso, profundidad_m], …] }         partes del cuerpo bajo el suelo
//
// La caja es un cuerpo dinámico con gravedad que descansa en el suelo (z = 0). Al agarrarla, una restricción fija la une
// a un cuerpo cinemático que sigue al agarre de las manos (MoveKinematic en cada subpaso): desde ahí la caja sigue a las
// manos como cuerpo físico (con su masa e inercia, sin teletransporte). Al soltarla, cae y se asienta.
// Las partes del cuerpo son cascos convexos cinemáticos que no empujan la caja: solo se consultan (CollideShape) para
// medir si algo la atraviesa o atraviesa el suelo.
import fs from 'node:fs';
import initJolt from 'jolt-physics';

const [, , entrada, salida] = process.argv;
const E = JSON.parse(fs.readFileSync(entrada, 'utf8'));
const J = await initJolt();

const FIJA = 0, CAJA = 1, HUESO = 2, CONSULTA = 3, AGARRE = 4, NL = 5;
const s = new J.JoltSettings();
const olf = new J.ObjectLayerPairFilterTable(NL);
olf.EnableCollision(FIJA, CAJA);
olf.EnableCollision(CONSULTA, HUESO); olf.EnableCollision(CONSULTA, FIJA);
const bpi = new J.BroadPhaseLayerInterfaceTable(NL, 2);
bpi.MapObjectToBroadPhaseLayer(FIJA, new J.BroadPhaseLayer(0));
for (const l of [CAJA, HUESO, CONSULTA, AGARRE]) bpi.MapObjectToBroadPhaseLayer(l, new J.BroadPhaseLayer(1));
s.mObjectLayerPairFilter = olf; s.mBroadPhaseLayerInterface = bpi;
s.mObjectVsBroadPhaseLayerFilter = new J.ObjectVsBroadPhaseLayerFilterTable(bpi, 2, olf, NL);
const iface = new J.JoltInterface(s);
const sys = iface.GetPhysicsSystem(), bi = sys.GetBodyInterface(), nq = sys.GetNarrowPhaseQuery();
sys.SetGravity(new J.Vec3(0, 0, -9.81));

const v3 = (a) => new J.Vec3(a[0], a[1], a[2]);
const r3 = (a) => new J.RVec3(a[0], a[1], a[2]);
const q4 = (a) => new J.Quat(a[0], a[1], a[2], a[3]);

const suelo = bi.CreateBody(new J.BodyCreationSettings(new J.BoxShape(new J.Vec3(20, 20, 1), 0.0), new J.RVec3(0, 0, -1),
    J.Quat.prototype.sIdentity(), J.EMotionType_Static, FIJA));
bi.AddBody(suelo.GetID(), J.EActivation_DontActivate);
const idSuelo = suelo.GetID().GetIndexAndSequenceNumber();

const [w, d, h] = E.caja.medidas;
const formaCaja = new J.BoxShape(new J.Vec3(w / 2, d / 2, h / 2), 0.0);
const bcs = new J.BodyCreationSettings(formaCaja, r3(E.caja.pose0), q4(E.caja.pose0.slice(3)), J.EMotionType_Dynamic, CAJA);
bcs.mOverrideMassProperties = J.EOverrideMassProperties_CalculateInertia;
bcs.mMassPropertiesOverride.mMass = E.caja.masa;
bcs.mFriction = 0.6;
const caja = bi.CreateBody(bcs);
bi.AddBody(caja.GetID(), J.EActivation_Activate);

const m0 = E.mano[0];
const agarre = bi.CreateBody(new J.BodyCreationSettings(new J.SphereShape(0.02), r3(m0), q4(m0.slice(3)),
    J.EMotionType_Kinematic, AGARRE));
bi.AddBody(agarre.GetID(), J.EActivation_Activate);

const huesos = Object.keys(E.cascos), cuerpos = {}, porId = new Map();
for (const hn of huesos) {
    const hs = new J.ConvexHullShapeSettings();
    for (const p of E.cascos[hn]) hs.mPoints.push_back(v3(p));
    hs.mMaxConvexRadius = 0.0;
    const r = hs.Create();
    if (!r.IsValid()) { console.error('casco inválido', hn, r.GetError().c_str()); process.exit(2); }
    const b = bi.CreateBody(new J.BodyCreationSettings(r.Get(), new J.RVec3(0, 0, 0), J.Quat.prototype.sIdentity(),
        J.EMotionType_Kinematic, HUESO));
    bi.AddBody(b.GetID(), J.EActivation_DontActivate);
    cuerpos[hn] = b; porId.set(b.GetID().GetIndexAndSequenceNumber(), hn);
}

function quatM(m) { // matriz 3x3 por filas → (x, y, z, w)
    const [a, b, c, dd, e, f, g, hh, i] = m, t = a + e + i;
    let W, x, y, z;
    if (t > 0) { const S = Math.sqrt(t + 1) * 2; W = .25 * S; x = (hh - f) / S; y = (c - g) / S; z = (dd - b) / S; }
    else if (a > e && a > i) { const S = Math.sqrt(1 + a - e - i) * 2; W = (hh - f) / S; x = .25 * S; y = (b + dd) / S; z = (c + g) / S; }
    else if (e > i) { const S = Math.sqrt(1 + e - a - i) * 2; W = (c - g) / S; x = (b + dd) / S; y = .25 * S; z = (f + hh) / S; }
    else { const S = Math.sqrt(1 + i - a - e) * 2; W = (dd - b) / S; x = (c + g) / S; y = (f + hh) / S; z = .25 * S; }
    return new J.Quat(x, y, z, W);
}
function nlerp(a, b, u) { // poses [x,y,z,qx,qy,qz,qw]
    const dot = a[3] * b[3] + a[4] * b[4] + a[5] * b[5] + a[6] * b[6], sg = dot < 0 ? -1 : 1;
    const q = [0, 1, 2, 3].map(k => a[3 + k] * (1 - u) + sg * b[3 + k] * u), n = Math.hypot(...q);
    return [0, 1, 2].map(k => a[k] * (1 - u) + b[k] * u).concat(q.map(x => x / n));
}

const ajustes = new J.CollideShapeSettings();
ajustes.mMaxSeparationDistance = 0.0;
const bpf = new J.DefaultBroadPhaseLayerFilter(s.mObjectVsBroadPhaseLayerFilter, CONSULTA);
const olfQ = new J.DefaultObjectLayerFilter(olf, CONSULTA);
const shf = new J.ShapeFilter(), uno = new J.Vec3(1, 1, 1), cero = new J.RVec3(0, 0, 0);

const fps = E.fps, sub = E.sub || 8, dt = 1 / fps / sub;
const [desde, hasta] = E.agarre;
let restr = null;
const out = { caja: [], caja_cuerpo: [], suelo: [] };
// se deja asentar la caja medio segundo antes de empezar (si empieza en el suelo)
if (desde > 0) for (let k = 0; k < fps * sub / 2; k++) { bi.MoveKinematic(agarre.GetID(), r3(m0), q4(m0.slice(3)), dt); iface.Step(dt, 1); }
const T = E.mano.length;
for (let t = 0; t < T; t++) {
    if (t === desde && restr === null) {
        const fc = new J.FixedConstraintSettings();
        fc.mAutoDetectPoint = true;
        restr = fc.Create(agarre, caja);
        sys.AddConstraint(restr);
        bi.ActivateBody(caja.GetID());
    }
    if (hasta !== null && t === hasta && restr !== null) { sys.RemoveConstraint(restr); restr = null; bi.ActivateBody(caja.GetID()); }
    // la caja en este cuadro
    const p = bi.GetPosition(caja.GetID()), q = bi.GetRotation(caja.GetID());
    out.caja.push([p.GetX(), p.GetY(), p.GetZ(), q.GetX(), q.GetY(), q.GetZ(), q.GetW()]);
    // el cuerpo en este cuadro y sus consultas
    const cu = E.cuadros[t];
    for (const hn of huesos) {
        const m = cu[hn];
        bi.SetPositionAndRotation(cuerpos[hn].GetID(), new J.RVec3(m[9], m[10], m[11]), quatM(m.slice(0, 9)), J.EActivation_DontActivate);
    }
    {   // la caja contra las partes del cuerpo
        const col = new J.CollideShapeAllHitCollisionCollector();
        const bf = new J.IgnoreSingleBodyFilter(caja.GetID());
        nq.CollideShape(formaCaja, uno, caja.GetCenterOfMassTransform(), ajustes, cero, col, bpf, olfQ, bf, shf);
        for (let k = 0; k < col.mHits.size(); k++) {
            const r = col.mHits.at(k), otro = porId.get(r.mBodyID2.GetIndexAndSequenceNumber());
            if (otro && r.mPenetrationDepth > 0) out.caja_cuerpo.push([t, otro, r.mPenetrationDepth]);
        }
        J.destroy(col); J.destroy(bf);
    }
    for (const hn of huesos) {  // cada parte contra el suelo
        const b = cuerpos[hn], col = new J.CollideShapeAllHitCollisionCollector(), bf = new J.IgnoreSingleBodyFilter(b.GetID());
        nq.CollideShape(b.GetShape(), uno, b.GetCenterOfMassTransform(), ajustes, cero, col, bpf, olfQ, bf, shf);
        for (let k = 0; k < col.mHits.size(); k++) {
            const r = col.mHits.at(k);
            if (r.mBodyID2.GetIndexAndSequenceNumber() === idSuelo && r.mPenetrationDepth > 0) out.suelo.push([t, hn, r.mPenetrationDepth]);
        }
        J.destroy(col); J.destroy(bf);
    }
    // simular hasta el cuadro siguiente: el agarre cinemático va de su pose de este cuadro a la del siguiente
    if (t + 1 < T) {
        for (let k = 1; k <= sub; k++) {
            const g = nlerp(E.mano[t], E.mano[t + 1], k / sub);
            bi.MoveKinematic(agarre.GetID(), r3(g), q4(g.slice(3)), dt);
            iface.Step(dt, 1);
        }
    }
}
fs.writeFileSync(salida, JSON.stringify(out));
console.log(`jolt: ${T} cuadros; caja dentro del cuerpo en ${new Set(out.caja_cuerpo.map(x => x[0])).size} cuadros, partes bajo el suelo en ${new Set(out.suelo.map(x => x[0])).size}`);
