# AI Running Coach

Primeiro MVP do projeto de análise de treinos e planejamento de corrida.

## Etapa atual

- Importação da exportação do Strava
- Seleção das corridas, padronização de datas e números, cálculo de pace
- Feature engineering: carga de treino (proxy), ACWR, volume/frequência recentes, tendência de pace
- Modelo de ML (regressão de quantil) + Riegel personalizado pra prever tempo de prova
- Gerador de plano de treino periodizado (base/build/peak/taper) via CLI

## Estrutura

```text
AI-Running-Coach/
├── data/
│   ├── raw/
│   │   └── activities.csv
│   └── processed/
│       ├── running_data.csv
│       ├── running_features.csv
│       └── weekly_summary.csv
├── src/
│   ├── data_processing.py       # limpeza dos dados do Strava
│   ├── feature_engineering.py   # carga de treino, ACWR, volume, tendência
│   ├── performance_model.py     # previsão de tempo de prova (ML + Riegel)
│   ├── plan_generator.py        # geração do plano periodizado
│   └── main.py                  # CLI — ponto de entrada
├── output/                      # planos gerados (.csv e .txt)
└── requirements.txt
```

## Como usar

```bash
cd src
pip install -r ../requirements.txt

# plano de 12 semanas pra 10km
python main.py --distance 10k --weeks 12

# meia maratona com data de prova (calcula as semanas automaticamente)
python main.py --distance 21k --race-date 2026-12-20

# maratona com tempo-meta específico
python main.py --distance 42k --weeks 16 --goal-time 3:45:00
```

O plano completo (todas as semanas, com sessões dia-a-dia) é salvo em
`output/training_plan.csv` e `output/training_plan.txt`.

### Observações importantes

- **Corridas não-representativas** (ex: feitas lesionado) podem ser excluídas
  dos modelos editando `EXCLUDE_FROM_MODEL` em `src/feature_engineering.py`.
- A previsão de performance é uma estimativa baseada no seu histórico — trate
  como referência, não como garantia.
- O gerador de plano nunca aumenta o volume semanal mais rápido que ~10%
  (regra clássica de prevenção de lesão). Se seu volume atual está muito
  abaixo do ideal pra distância/prazo escolhidos, ele avisa em vez de forçar
  um salto perigoso.
