"""Saúde padronizada do robô (brief, seção 35): CPU, memória, acelerador, disco,
temperatura, bateria, câmeras, sensores, rede e sincronização — na telemetria."""

from __future__ import annotations

from pathlib import Path

from conftest import write_images

from valtexa_rover.hal.interfaces import HardwareUnavailableError, SystemMonitor
from valtexa_rover.hal.local import LinuxSystemMonitor
from valtexa_rover.hal.simulated import FolderCamera, SimulatedSystemMonitor


def _proc(raiz: Path, *, user: int, idle: int, mem_total_kb: int = 8_000_000,
          disponivel_kb: int = 6_000_000, carga: str = "0.52") -> Path:
    raiz.mkdir(parents=True, exist_ok=True)
    (raiz / "stat").write_text(f"cpu  {user} 0 0 {idle} 0 0 0 0 0 0\ncpu0 1 0 0 1 0 0 0 0\n")
    (raiz / "meminfo").write_text(
        f"MemTotal:       {mem_total_kb} kB\nMemFree:          100000 kB\n"
        f"MemAvailable:   {disponivel_kb} kB\n")
    (raiz / "loadavg").write_text(f"{carga} 0.40 0.30 1/200 1234\n")
    return raiz


def test_monitor_linux_le_cpu_memoria_e_carga_do_proc(tmp_path) -> None:
    proc = _proc(tmp_path / "proc", user=100, idle=900)
    monitor = LinuxSystemMonitor(proc)
    assert isinstance(monitor, SystemMonitor)
    assert monitor.describe().simulated is False
    primeira = monitor.read()
    assert primeira.cpu_percent is None  # sem referência ainda: não inventa
    assert primeira.memory_total_bytes == 8_000_000 * 1024
    assert primeira.memory_used_bytes == 2_000_000 * 1024
    assert primeira.load_1m == 0.52
    assert primeira.accelerator_percent is None  # depende do fabricante: não medido
    _proc(proc, user=400, idle=1000)  # +300 ocupado, +100 ocioso
    assert monitor.read().cpu_percent == 75.0


def test_monitor_linux_sem_proc_devolve_nada_medido(tmp_path) -> None:
    leitura = LinuxSystemMonitor(tmp_path / "nao-existe").read()
    assert (leitura.cpu_percent, leitura.memory_used_bytes, leitura.load_1m) == (None, None, None)


def test_monitor_real_le_o_proc_desta_maquina() -> None:
    if not Path("/proc/meminfo").exists():
        return  # fora do Linux não há o que ler
    leitura = LinuxSystemMonitor().read()
    assert leitura.memory_total_bytes and leitura.memory_total_bytes > 0


def test_telemetria_leva_saude_padronizada(rig_factory) -> None:
    rig = rig_factory(["verde"] * 12, telemetry_interval_s=5.0)
    rig.hardware.system = SimulatedSystemMonitor(cpu_percent=81.0)
    rig.core = rig.new_core()
    rig.core.start_mission("m-1", "talhao-1")
    rig.run(12)
    rig.core.sync.run_until_idle()
    amostras = sorted(rig.cloud.telemetry.values(), key=lambda a: a["sequence_number"])
    saude = amostras[-1]["health"]
    assert saude["cpu_percent"] == 81.0 and saude["memory_percent"] == 37.5
    assert saude["accelerator"]["percent"] == 40.0
    assert saude["accelerator"]["providers"]  # o ONNX Runtime de verdade
    assert 0 < saude["disk_percent"] < 100
    assert saude["cameras"] == [{"id": "cam-0", "ok": True, "detail": None}]
    assert saude["sensors"] == {"gps": "ok", "imu": "nao_monitorado", "ambiente": "ok"}
    assert saude["network"] == {"online": True}
    assert saude["edge_ai"]["model_loaded"] is True
    assert saude["sync"]["pending"] >= 0 and saude["sync"]["blocked"] is False
    # o que é simulação vem dito
    assert {"sistema", "camera", "sensores"} <= set(saude["simulated"])


def test_saude_sem_monitor_de_sistema_diz_que_nao_mediu(rig_factory) -> None:
    rig = rig_factory(["verde"] * 2)
    saude = rig.core.health.relatorio(online=False)
    assert saude["cpu_percent"] is None and saude["memory_percent"] is None
    assert saude["network"] == {"online": False}


def test_camera_com_falha_e_fila_parada_aparecem_na_saude(rig_factory, tmp_path) -> None:
    rig = rig_factory(["verde"] * 4)
    pasta = write_images(tmp_path / "img2", ["verde"] * 4)
    rig.hardware.cameras = [FolderCamera("cam-0", pasta, rig.clock, fail_on=[1, 2])]
    rig.core.start_mission("m-1", "talhao-1")
    rig.network.set_online(False)
    rig.run(2)
    saude = rig.core.health.relatorio(online=False)
    assert saude["cameras"][0]["ok"] is False and saude["cameras"][0]["detail"]
    rig.clock.advance(600)
    fila = rig.core.health.relatorio(online=False)["sync"]
    assert fila["pending"] > 0  # sem rede, a fila só cresce
    assert fila["oldest_pending_s"] >= 600


def test_monitor_que_falha_vira_falha_ativa(rig_factory) -> None:
    class Quebrado:
        def read(self):
            raise HardwareUnavailableError("sensor de sistema não respondeu")

        def describe(self):
            return SimulatedSystemMonitor().describe()

    rig = rig_factory(["verde"])
    rig.hardware.system = Quebrado()
    saude = rig.core.health.relatorio(online=True)
    assert saude["cpu_percent"] is None and "system" in rig.core.health.faults
