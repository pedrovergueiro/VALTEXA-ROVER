from __future__ import annotations

import json

from valtexa_rover.cli import main


def test_config_padrao_e_validacao(tmp_path, capsys) -> None:
    assert main(["config-padrao"]) == 0
    padrao = json.loads(capsys.readouterr().out)
    caminho = tmp_path / "rover.json"
    caminho.write_text(json.dumps(padrao))
    assert main(["validar-config", str(caminho)]) == 0
    caminho.write_text(json.dumps({"retencao_errada": {}}))
    assert main(["validar-config", str(caminho)]) == 1


def test_status_do_banco_local(rig_factory, capsys) -> None:
    rig = rig_factory(["vermelho", "verde"], auto_sync=False)
    rig.core.start_mission("m-1", "talhao-1")
    rig.run(2)
    rig.core.close()
    capsys.readouterr()
    assert main(["status", "--dados", str(rig.data_dir)]) == 0
    saida = json.loads(capsys.readouterr().out)
    assert saida["last_mission"]["id"] == "m-1" and saida["last_mission"]["frames_done"] == 2
    assert saida["active_model"]["sha256"] == rig.sha
    assert any(item["status"] == "PENDENTE" for item in saida["queue"])
