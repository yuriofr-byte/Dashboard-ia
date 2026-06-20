import pandas as pd
import matplotlib.pyplot as plt
import requests

ARQUIVO_CSV = "vendas.csv"
import os
API_KEY = os.environ.get("GEMINI_API_KEY", "")


def carregar_dados(arquivo):
    try:
        df = pd.read_csv(arquivo)
        print(f"Dados carregados! {len(df)} registos encontrados.")
        return df
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None


def resumo_dados(df):
    print("\nRESUMO DOS DADOS:")
    print("-" * 40)
    print(df.describe())
    print("-" * 40)


def gerar_graficos(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Dashboard de Vendas", fontsize=16)

    vendas_mes = df.groupby("mes")["vendas"].sum()
    axes[0, 0].bar(vendas_mes.index, vendas_mes.values, color="steelblue")
    axes[0, 0].set_title("Vendas por Mes")
    axes[0, 0].set_xlabel("Mes")
    axes[0, 0].set_ylabel("Vendas")

    lucro_mes = df.groupby("mes")["lucro"].sum()
    axes[0, 1].plot(lucro_mes.index, lucro_mes.values, color="green", marker="o")
    axes[0, 1].set_title("Lucro por Mes")
    axes[0, 1].set_xlabel("Mes")
    axes[0, 1].set_ylabel("Lucro")

    vendas_produto = df.groupby("produto")["vendas"].sum()
    axes[1, 0].pie(vendas_produto.values, labels=vendas_produto.index, autopct="%1.1f%%")
    axes[1, 0].set_title("Vendas por Produto")

    axes[1, 1].scatter(df["vendas"], df["lucro"], color="purple", alpha=0.6)
    axes[1, 1].set_title("Vendas vs Lucro")
    axes[1, 1].set_xlabel("Vendas")
    axes[1, 1].set_ylabel("Lucro")

    plt.tight_layout()
    plt.savefig("dashboard.png")
    print("Grafico salvo como dashboard.png!")
    plt.show()


def perguntar_ia(df, pergunta):
    print(f"\nPerguntando a IA: {pergunta}")

    resumo = df.describe().to_string()
    dados = df.to_string()

    prompt = f"""
Tens acesso a um dataset de vendas com os seguintes dados:

RESUMO ESTATISTICO:
{resumo}

DADOS COMPLETOS:
{dados}

Pergunta: {pergunta}

Responde de forma clara e objetiva em portugues.
"""

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"
        response = requests.post(
            url,
            json={"contents": [{"parts": [{"text": prompt}]}]}
        )
        resultado = response.json()
        texto = resultado["candidates"][0]["content"]["parts"][0]["text"]
        print(f"\nIA: {texto}")

    except Exception as e:
        print(f"Erro na IA: {e}")
        try:
            print(f"Resposta: {response.json()}")
        except:
            pass


if __name__ == "__main__":
    print("Dashboard de Analise de Dados com IA")
    print("=" * 40)

    df = carregar_dados(ARQUIVO_CSV)

    if df is not None:
        resumo_dados(df)
        gerar_graficos(df)

        print("\nFaz uma pergunta sobre os dados!")
        pergunta = input("\nA tua pergunta: ")
        perguntar_ia(df, pergunta)