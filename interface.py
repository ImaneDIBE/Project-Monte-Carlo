import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm

st.title('Monte carlo')

def section_loi_normale():
    st.header('Simulation de la loi normale')
    
    # Ajouter le widget 
    m = st.number_input('Entrez la moyenne')
    std = st.number_input('Entrez l ecart type')
    n = st.number_input('Entrez le nombre d echantillon aleatoires')

    m = int(m)
    std = int(std)
    n = int(n)

    def monte_carlo_normal(m, std, n):
        samples = np.random.normal(m, std, n)
        return samples

    if st.button('Afficher le graphe'):
        # Simulation
        samples = monte_carlo_normal(m, std, n)

        # Affichage de l'histogramme des échantillons
        plt.hist(samples, bins=50, density=True, color='blue', edgecolor='black', alpha=0.7)
        plt.xlabel('Valeurs')
        plt.ylabel('Densité de probabilité')
        plt.title('Simulation de loi normale avec Monte Carlo')
        plt.grid(True)
        st.pyplot(plt)

def section_mouvement_brownien():
    st.header('Mouvement Brownien Standard')
    
    # Ajouter les widget 
    dt = st.number_input('Entrez l intervalle du temps')
    n = st.number_input('Entrez n')
    p = st.number_input('Entrez le pas nombre des trajectoires des mouvements')

    n = int(n)
    p = int(p)

    def standard_brownian_motion(dt, n, p):
        t = np.linspace(0, n*dt, num=n)
        W = np.random.standard_normal(size=(p, n))
        W = np.cumsum(W, axis=1) * np.sqrt(dt)  # Brownian motion
        return t, W

    if st.button('Afficher le graphe '):
        # Simulation du mouvement brownien standard
        t, W = standard_brownian_motion(dt, n, p)

        # Affichage des trajectoires
        for i in range(p):
            plt.plot(t, W[i])
        plt.xlabel('Temps')
        plt.ylabel('Position')
        plt.title('Mouvement Brownien Standard')
        plt.grid(True)
        st.pyplot(plt)

def section_mouvement_brownien_geometrique():
    st.header('Mouvement Brownien Geometrique')
    
    mu = st.number_input('Entrez le taux de rendement moyen de l actif')
    sigma = st.number_input('Entrez la volatilite de l actif')
    S0 = st.number_input('Entrez le prix de l actif')
    dt = st.number_input(' Entrez l intervalle du temps')
    n = st.number_input(' Entrez n')
    p = st.number_input(' Entrez le pas nombre des trajectoires des mouvements')

    n = int(n)
    p = int(p)

    # Fonction pour simuler un mouvement brownien géométrique
    def brownien_geometrique(mu, sigma, S0, dt, n, p):
        t = np.linspace(0, n*dt, num=n)
        W = np.random.standard_normal(size=(p , n))
        W = np.cumsum(W, axis=1) * np.sqrt(dt)  # Mouvement Brownien 
        X = (mu - 0.5 * sigma**2) * t + sigma * W
        S = S0 * np.exp(X)  # Mouvement Brownien Geometrique
        return t, S

    if st.button(' Afficher le graphe'):
        # Simulation du mouvement brownien géométrique
        t, S = brownien_geometrique(mu, sigma, S0, dt, n, p)

        # Affichage des trajectoires
        for i in range(p):
            plt.plot(t, S[i])
        plt.xlabel('Temps')
        plt.ylabel('Prix')
        plt.title('Mouvement Brownien Géométrique')
        st.pyplot(plt)

def section_mouvement_brownien_arithmetique():
    st.header('Mouvement Brownien arithmetique')
    
    mu = st.number_input('Entrez le taux de rendement moyen de l actif ')
    sigma = st.number_input('Entrez la volatilite de l actif ')
    S0 = st.number_input('Entrez le prix de l actif ')
    dt = st.number_input(' Entrez l intervalle du temps ')
    n = st.number_input(' Entrez n ')
    p = st.number_input(' Entrez le pas nombre des trajectoires des mouvements ')

    n = int(n)
    p = int(p)

    # Fonction pour simuler un mouvement brownien arithmétique
    def brownien_arithmetique(mu, sigma, S0, dt, n, p):
        t = np.linspace(0, n*dt, num=n)
        W = np.random.standard_normal(size=(p, n))
        W = np.cumsum(W, axis=1) * np.sqrt(dt)  # Mouvement Brownien
        X = (mu - 0.5 * sigma**2) * t + sigma * W
        S = S0 + X  # Mouvement Brownien Arithmetic 
        return t, S

    if st.button(' Afficher le graphe '):
        # Simulation du mouvement brownien arithmétique
        t, S = brownien_arithmetique(mu, sigma, S0, dt, n, p)

        # Affichage des trajectoires
        for i in range(p):
            plt.plot(t, S[i])
        plt.xlabel('Temps')
        plt.ylabel('Prix')
        plt.title('Mouvement Brownien arithmétiquee')
        st.pyplot(plt)

def section_evaluation_options():
    st.header('Evaluation des options européennes')

    option_type = st.text_input('Sélectionnez le type de l option (call ou put)')
    ticker = st.text_input('Sélectionnez le ticker de l option (ex: AAPL)')
    start = st.date_input("Choisissez la date de début")
    end = st.date_input("Choisissez la date de fin")
    K = st.number_input('Entrez le prix d exercice de l option')
    T = st.number_input('Entrez le temps jusqu à l expiration de l option (en années)')
    r = st.number_input('Entrez le taux d intérêt sans risque')

    def black_scholes_option_price(S, K, T, r, sigma, option_type):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == 'put':
            option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("Option type should be 'call' or 'put'.")

        return option_price

    try:
        # Téléchargement des données à partir de Yahoo Finance
        data = yf.download(ticker, start, end)

        # Calcul des rendements log-normaux pour estimer la volatilité historique
        returns = np.log(data['Close'] / data['Close'].shift(1))
        sigma = returns.std() * np.sqrt(252)  # Volatilité historique annualisée
        S = data['Close'].iloc[-1]  # Prix actuel de l'actif sous-jacent

        if st.button(' Afficher le graphe  '):
            # Générer une plage de prix sous-jacents pour visualiser la relation prix de l'option / prix sous-jacent
            underlying_prices = np.linspace(S * 0.8, S * 1.2, 100)
            option_prices = [black_scholes_option_price(price, K, T, r, sigma, option_type) for price in underlying_prices]

            # Visualisation
            plt.plot(underlying_prices, option_prices, label="Prix de l'option européenne")
            plt.axvline(x=S, color='r', linestyle='--', label='Prix actuel de l actif sous-jacent')
            plt.xlabel('Prix de l actif sous-jacent')
            plt.ylabel('Prix de l option')
            plt.title('Prix de l option européenne selon le modèle Black-Scholes')
            plt.legend()
            st.pyplot(plt)

    except Exception as e:
        st.error(f"Une erreur s'est produite lors du téléchargement des données : {str(e)}")


# Créer le menu latéral
st.sidebar.title('Menu')

# Ajouter les liens vers les différentes sections
section_selected = st.sidebar.radio('Naviguer vers :', 
                                    ('Loi Normale', 'Mouvement Brownien', 
                                     'Mouvement Brownien Geometrique', 
                                     'Mouvement Brownien Arithmetique', 
                                     'Evaluation des Options'))

# Appeler la fonction appropriée en fonction de la sélection de l'utilisateur
if section_selected == 'Loi Normale':
    section_loi_normale()
elif section_selected == 'Mouvement Brownien':
    section_mouvement_brownien()
elif section_selected == 'Mouvement Brownien Geometrique':
    section_mouvement_brownien_geometrique()
elif section_selected == 'Mouvement Brownien Arithmetique':
    section_mouvement_brownien_arithmetique()
elif section_selected == 'Evaluation des Options':
    section_evaluation_options()

