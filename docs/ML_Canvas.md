# OWNML MACHINE LEARNING CANVAS

**Designed for:** Ray Santos  
**Designed by:** Amanda Silveira  
**Date:** 19/04/2026  
**Iteration:** 1.0  

---

## VALUE PROPOSITION
Reduzir churn e proteger a Receita Recorrente Mensal (Monthly Recurring Revenue - MRR) por meio de um modelo preditivo que identifica antecipadamente clientes com alto risco de cancelamento, permitindo ações proativas das equipes de relacionamento com clientes, marketing e liderança, com base no risco individual de cada conta. O impacto final é o aumento do Lifetime Value (LTV).

## PREDICTION TASK
Classificação Binária.  
**Input:** Um perfil de cliente (dados financeiros, contratuais e comportamentais).  
**Output:** Probabilidade [0 a 1] de Churn nos próximos 30 dias.

## DECISIONS
Se P(Churn) > Limiar (ex: 70%), a conta ingressa em uma lista prioritária de retenção onde um analista oferece desconto ou melhoria de plano.  
Se P(Churn) < Limiar, a conta segue o fluxo normal (não gastamos capital promocional).

## DATA SOURCES
IBM Telco Customer Churn dataset, banco de dados relacional e CRM utilizado principalmente para estudos e pesquisa. Contém tabelas de demografia, serviços de conta e faturamento mensal.  
Link: https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset/data

## DATA COLLECTION
**Input:** Features geradas pelo cruzamento financeiro vs dados da contratação vs dados comportamentais num período estático. Se houver: Tratamento de dados faltantes/categóricos (Adicionar nome das colunas que serão tratadas).

## FEATURES
- **Demográficos:** Gender, Age, Senior Citizen, Married, Dependents, Number of Dependents
- **Localização (agregada):** State, City, Zip Code
- **Serviços e Contrato:** Phone Service, Multiple Lines, Internet Service, Contract, Offer, Paperless Billing, Payment Method
- **Uso e Engajamento:** Tenure in Months, Avg Monthly GB Download, Streaming TV, Streaming Movies, Streaming Music, Online Security, Online Backup, Device Protection Plan, Premium Tech Support, Unlimited Data, Referred a Friend, Number of Referrals
- **Financeiro:** Monthly Charge, Total Charges, Total Refunds, Total Extra Data Charges, Total Long Distance Charges
- **Satisfação e Valor:** Satisfaction Score, CLTV

## BUILDING MODELS
Treinaremos uma Rede Neural (MLP em PyTorch) usando busca de parâmetros. Avaliaremos frente ao baseline (Scikit-Learn). Usaremos o MLFlow para registrar os experimentos.  
O modelo deve ser executado mensalmente para avaliação do cenário atual de churn da empresa.

## MAKING PREDICTIONS
**Quando:** Via API em Real-Time e em Batch automatizado mensalmente para gerar a lista de retenção baseada em todos os clientes.  
**Requisito:** A inferência via API deve ocorrer com latência < 200ms.

## IMPACT SIMULATION
- **Verdadeiro Positivo (Acerto na retenção):** Ticket médio mantido descontando o custo da oferta.  
- **Falso Positivo (Promoção desperdiçada):** Dinheiro gasto sem necessidade.  
- **Falso Negativo (Cliente e Receita perdidos):** Pior cenário: modelo errou e perdemos o Lifetime Value/Mensalidade.  
- **Verdadeiro Negativo:** R$ 0 Nenhuma ação/gasto necessário.

## MONITORING
Monitorar ativamente o percentual de clientes que foram previstos como "Alto Risco", receberam desconto, mas ainda assim cancelaram contra o grupo que não recebeu desconto.  
**Métricas de Saúde:** Rastreador de Drifting de Dados (ver se a demografia está mudando com o tempo) e latência da infraestrutura FastAPI na nuvem AWS.
