"""IMU parada: módulo da gravidade, ruído e viés do giroscópio, eixos coerentes.

Com o robô **parado** e a inclinação conhecida (nível de bolha ou inclinômetro no chassi:
``roll_esperado_deg``/``pitch_esperado_deg``), o ensaio lê a IMU a ``taxa_hz`` por
``duracao_s`` e confere:

* **gravidade**: a média de ``|a|`` perto de 9,81 m/s² (escala do acelerômetro certa);
* **inclinação**: a rolagem e a arfagem que o sensor publica perto das informadas;
* **eixos**: a inclinação que a própria gravidade medida implica igual à publicada — se
  o MTi estiver montado girado em relação ao robô (ou o ID/escala CAN trocado), os dois
  discordam;
* **giroscópio**: ruído (desvio) e viés (média) da velocidade angular parada. Sem o
  giroscópio no CAN (``gyro_id``), a taxa sai da derivada dos ângulos — o laudo diz.
"""

from __future__ import annotations

import math
from typing import Any

from valtexa_rover.ensaios.base import (
    Contexto,
    Ensaio,
    EnsaioIndisponivelError,
    Param,
    criterio,
)
from valtexa_rover.ensaios.estatistica import (
    GRAVIDADE_PADRAO,
    desvio,
    diferenca_angular,
    inclinacao_pela_gravidade,
    media,
    taxas,
)


class Imu(Ensaio):
    nome = "imu"
    fase = 0
    titulo = "IMU parada: gravidade, giroscópio e eixos"
    parametros = (
        Param("duracao_s", float, 30.0, 1.0, 3600.0, "leitura parada (s)"),
        Param("taxa_hz", float, 50.0, 1.0, 400.0, "leituras por segundo"),
        Param("roll_esperado_deg", float, 0.0, -90.0, 90.0, "rolagem informada (°)"),
        Param("pitch_esperado_deg", float, 0.0, -90.0, 90.0, "arfagem informada (°)"),
        Param("tolerancia_g_ms2", float, 0.1, 0.0, 5.0, "erro máximo de |a| (m/s²)"),
        Param("tolerancia_inclinacao_deg", float, 1.5, 0.0, 45.0,
              "erro máximo de rolagem/arfagem (°)"),
        Param("ruido_gyro_max_dps", float, 0.2, 0.0, 100.0, "ruído máximo do giroscópio (°/s)"),
        Param("vies_gyro_max_dps", float, 0.5, 0.0, 100.0, "viés máximo do giroscópio (°/s)"),
    )
    hardware_real = ("simulado/não testado em hardware: IDs e escalas CAN do MTi-630 a "
                     "conferir no MT Manager; giroscópio só com gyro_id configurado.")

    def executar(self, ctx: Contexto, p: dict[str, Any]) -> dict[str, Any]:
        if ctx.hardware.imu is None:
            raise EnsaioIndisponivelError("rover sem IMU (hardware.json: imu)")
        leituras = max(2, round(p["duracao_s"] * p["taxa_hz"]))
        intervalo = 1.0 / p["taxa_hz"]
        t, roll, pitch, yaw, gmod, roll_acc, pitch_acc = [], [], [], [], [], [], []
        acc: list[list[float]] = [[], [], []]
        gyro: list[tuple[float, float, float]] = []
        falhas = 0
        inicio = ctx.clock.monotonic()
        for i in range(leituras):
            try:
                r = ctx.hardware.imu.read()
            except Exception:  # leitura perdida conta, não para o ensaio
                falhas += 1
                r = None
            if r is not None:
                ax, ay, az = r.accel_ms2
                t.append(ctx.clock.monotonic() - inicio)
                roll.append(r.roll_deg)
                pitch.append(r.pitch_deg)
                yaw.append(r.yaw_deg)
                for k, v in enumerate((ax, ay, az)):
                    acc[k].append(v)
                gmod.append(math.sqrt(ax * ax + ay * ay + az * az))
                ra, pa = inclinacao_pela_gravidade(ax, ay, az)
                roll_acc.append(ra)
                pitch_acc.append(pa)
                if r.gyro_dps is not None:
                    gyro.append(tuple(r.gyro_dps))
            if i % max(1, leituras // 20) == 0:
                ctx.avisa((i + 1) / leituras, f"leitura {i + 1}/{leituras}")
            ctx.espera(intervalo)
        if len(t) < 2:
            raise EnsaioIndisponivelError(f"IMU não respondeu ({falhas} falhas)")
        if len(gyro) >= 2:
            fonte = "giroscópio"
            eixos = [[g[k] for g in gyro] for k in range(3)]
        else:
            fonte = "derivada dos ângulos (sem gyro_id no hardware.json)"
            eixos = [taxas(roll, t), taxas(pitch, t), taxas(yaw, t)]
        ruido = max(desvio(e) for e in eixos)
        vies = max(abs(media(e)) for e in eixos if e) if any(eixos) else None
        g = media(gmod)
        r_med, p_med = media(roll), media(pitch)
        ra_med, pa_med = media(roll_acc), media(pitch_acc)
        tol = p["tolerancia_inclinacao_deg"]
        criterios = [
            criterio("erro_gravidade_ms2", abs(g - GRAVIDADE_PADRAO), p["tolerancia_g_ms2"],
                     "<=", "m/s²"),
            criterio("erro_roll_deg", abs(diferenca_angular(r_med, p["roll_esperado_deg"])),
                     tol, "<=", "°"),
            criterio("erro_pitch_deg", abs(diferenca_angular(p_med, p["pitch_esperado_deg"])),
                     tol, "<=", "°"),
            criterio("eixos_roll_deg", abs(diferenca_angular(ra_med, r_med)), tol, "<=", "°"),
            criterio("eixos_pitch_deg", abs(diferenca_angular(pa_med, p_med)), tol, "<=", "°"),
            criterio("ruido_gyro_dps", ruido, p["ruido_gyro_max_dps"], "<=", "°/s"),
            criterio("vies_gyro_dps", vies, p["vies_gyro_max_dps"], "<=", "°/s"),
        ]
        medidas = {
            "gravidade_ms2": round(g, 4), "gravidade_desvio_ms2": round(desvio(gmod), 4),
            "roll_deg": round(r_med, 3), "pitch_deg": round(p_med, 3),
            "roll_pela_gravidade_deg": round(ra_med, 3),
            "pitch_pela_gravidade_deg": round(pa_med, 3),
            "ruido_gyro_dps": round(ruido, 4), "vies_gyro_dps": None if vies is None
            else round(vies, 4), "fonte_gyro": fonte, "leituras": len(t), "falhas": falhas,
        }
        # Brutos: a plataforma refaz |a| e a inclinação pela gravidade com eles.
        amostras = {"roll_deg": roll[-20000:], "pitch_deg": pitch[-20000:],
                    "accel_ms2": [[round(v, 5) for v in eixo[-20000:]] for eixo in acc],
                    "gyro_dps": [[round(v, 5) for v in e[-20000:]] for e in eixos],
                    "fonte_gyro": fonte,
                    "esperado": [p["roll_esperado_deg"], p["pitch_esperado_deg"]]}
        return {"criterios": criterios, "medidas": medidas, "amostras": amostras}
