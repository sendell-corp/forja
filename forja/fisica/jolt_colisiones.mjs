// Forja · fisica/jolt_colisiones.mjs — colisiones de la criatura cuadro a cuadro con Jolt (jolt-physics 1.1.1, MIT).
//
//   node forja/fisica/jolt_colisiones.mjs ENTRADA.json SALIDA.json
//
// ENTRADA: { "cascos": {hueso: [[x,y,z], …]}   puntos del casco en la pose de reposo (coordenadas del rig, Z arriba)
//            "cuadros": [{hueso: [r00,r01,r02,r10,r11,r12,r20,r21,r22,tx,ty,tz]}, …]   reposo → pose (rígido)
//            "pares": [[a, b], …]   pares de huesos que no deben tocarse; "suelo": true }
// SALIDA:  { "suelo": [[cuadro, hueso, profundidad_m, [x,y,z]], …],
//            "pares": [[cuadro, a, b, profundidad_m, [eje x,y,z], [x,y,z]], …] }
//
// Cada hueso es un cuerpo cinemático con un ConvexHullShape de su parte de la malla; el suelo, una caja estática con la
// cara de arriba en z = 0. En cada cuadro se colocan los cuerpos y se pregunta a la fase estrecha de Jolt
// (NarrowPhaseQuery.CollideShape) qué toca cada casco y cuánto penetra.
import fs from 'node:fs';
import initJolt from 'jolt-physics';

const [, , entrada, salida] = process.argv;
const E = JSON.parse(fs.readFileSync(entrada, 'utf8'));
const J = await initJolt();

const FIJA = 0, MOVIL = 1;
const s = new J.JoltSettings();
const olf = new J.ObjectLayerPairFilterTable(2);
olf.EnableCollision(FIJA, MOVIL); olf.EnableCollision(MOVIL, MOVIL);
const bpi = new J.BroadPhaseLayerInterfaceTable(2, 2);
bpi.MapObjectToBroadPhaseLayer(FIJA, new J.BroadPhaseLayer(0));
bpi.MapObjectToBroadPhaseLayer(MOVIL, new J.BroadPhaseLayer(1));
s.mObjectLayerPairFilter = olf; s.mBroadPhaseLayerInterface = bpi;
s.mObjectVsBroadPhaseLayerFilter = new J.ObjectVsBroadPhaseLayerFilterTable(bpi, 2, olf, 2);
const iface = new J.JoltInterface(s);
const sys = iface.GetPhysicsSystem(), bi = sys.GetBodyInterface(), nq = sys.GetNarrowPhaseQuery();

// suelo: caja de 40 x 40 m con la cara de arriba en z = 0
const suelo = bi.CreateBody(new J.BodyCreationSettings(new J.BoxShape(new J.Vec3(20, 20, 1), 0.0), new J.RVec3(0, 0, -1),
    J.Quat.prototype.sIdentity(), J.EMotionType_Static, FIJA));
bi.AddBody(suelo.GetID(), J.EActivation_DontActivate);

const huesos = Object.keys(E.cascos);
const cuerpos = {}, porId = new Map([[suelo.GetID().GetIndexAndSequenceNumber(), '__suelo__']]);
for (const h of huesos) {
    const hs = new J.ConvexHullShapeSettings();
    for (const [x, y, z] of E.cascos[h]) hs.mPoints.push_back(new J.Vec3(x, y, z));
    hs.mMaxConvexRadius = 0.0;
    const r = hs.Create();
    if (!r.IsValid()) { console.error('casco inválido', h, r.GetError().c_str()); process.exit(2); }
    const b = bi.CreateBody(new J.BodyCreationSettings(r.Get(), new J.RVec3(0, 0, 0), J.Quat.prototype.sIdentity(),
        J.EMotionType_Kinematic, MOVIL));
    bi.AddBody(b.GetID(), J.EActivation_DontActivate);
    cuerpos[h] = b; porId.set(b.GetID().GetIndexAndSequenceNumber(), h);
}
const pares = new Set((E.pares || []).map(([a, b]) => a < b ? `${a}|${b}` : `${b}|${a}`));

function quat(m) { // matriz 3x3 por filas → cuaternión (x, y, z, w)
    const [a, b, c, d, e, f, g, h, i] = m, t = a + e + i;
    let w, x, y, z;
    if (t > 0) { const S = Math.sqrt(t + 1) * 2; w = .25 * S; x = (h - f) / S; y = (c - g) / S; z = (d - b) / S; }
    else if (a > e && a > i) { const S = Math.sqrt(1 + a - e - i) * 2; w = (h - f) / S; x = .25 * S; y = (b + d) / S; z = (c + g) / S; }
    else if (e > i) { const S = Math.sqrt(1 + e - a - i) * 2; w = (c - g) / S; x = (b + d) / S; y = .25 * S; z = (f + h) / S; }
    else { const S = Math.sqrt(1 + i - a - e) * 2; w = (d - b) / S; x = (c + g) / S; y = (f + h) / S; z = .25 * S; }
    return new J.Quat(x, y, z, w);
}

const ajustes = new J.CollideShapeSettings();
const SEP = E.separacion || 0.0;  // > 0: cuenta también lo que está a menos de SEP (profundidad negativa)
ajustes.mMaxSeparationDistance = SEP;
const bpf = new J.DefaultBroadPhaseLayerFilter(s.mObjectVsBroadPhaseLayerFilter, MOVIL);
const olfF = new J.DefaultObjectLayerFilter(olf, MOVIL);
const shf = new J.ShapeFilter();
const uno = new J.Vec3(1, 1, 1), cero = new J.RVec3(0, 0, 0);
const out = { suelo: [], pares: [] };
E.cuadros.forEach((cuadro, t) => {
    for (const h of huesos) {
        const m = cuadro[h];
        bi.SetPositionAndRotation(cuerpos[h].GetID(), new J.RVec3(m[9], m[10], m[11]), quat(m.slice(0, 9)), J.EActivation_DontActivate);
    }
    for (const h of huesos) {
        const b = cuerpos[h];
        const col = new J.CollideShapeAllHitCollisionCollector();
        const bf = new J.IgnoreSingleBodyFilter(b.GetID());
        nq.CollideShape(b.GetShape(), uno, b.GetCenterOfMassTransform(), ajustes, cero, col, bpf, olfF, bf, shf);
        const hits = col.mHits;
        for (let k = 0; k < hits.size(); k++) {
            const r = hits.at(k), otro = porId.get(r.mBodyID2.GetIndexAndSequenceNumber()), d = r.mPenetrationDepth;
            const p = r.mContactPointOn1, ax = r.mPenetrationAxis;
            if (otro === '__suelo__') { if (d > -SEP) out.suelo.push([t, h, d, [p.GetX(), p.GetY(), p.GetZ()]]); }
            else if (h < otro && pares.has(`${h}|${otro}`) && d > -SEP)
                out.pares.push([t, h, otro, d, [ax.GetX(), ax.GetY(), ax.GetZ()], [p.GetX(), p.GetY(), p.GetZ()]]);
        }
        J.destroy(col); J.destroy(bf);
    }
});
fs.writeFileSync(salida, JSON.stringify(out));
console.log(`jolt: ${E.cuadros.length} cuadros, ${huesos.length} cascos; contactos con el suelo ${out.suelo.length}, entre partes ${out.pares.length}`);
