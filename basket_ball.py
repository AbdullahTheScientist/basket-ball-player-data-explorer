import pandas as pd
import numpy as np
import seaborn as sns
import base64
import streamlit as st
import matplotlib.pyplot as plt

st.title("NBA Player Stats Explorer")
st.write("This app performs simple web scraping of NBA Player stats.")
st.sidebar.header("User input features")
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950, 2024))))

# Web scraping of NBA player stats
@st.cache_data
def load_data(year):
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
    html = pd.read_html(url, header=0)
    df = html[0]
    raw = df.drop(df[df.Age == 'Age'].index)
    raw = raw.fillna(0)
    players_stat = raw.drop(['Rk'], axis=1)
    return players_stat

players_stat = load_data(selected_year)

# Sidebar team selection
sorted_unique_team = sorted(players_stat.Tm.unique())
selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)

# Sidebar - Position selection
unique_pos = ['C', 'PF', 'SF', 'PG', 'SG']
selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)

# Filtering data
df_selected_team = players_stat[(players_stat.Tm.isin(selected_team)) & (players_stat.Pos.isin(selected_pos))]

st.header('Display Player Stats of Selected Team(s)')
st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + ' rows and ' + str(df_selected_team.shape[1]) + ' columns.')
st.dataframe(df_selected_team)

def file_downloaded(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="players_stat.csv">Download CSV File</a>' 
    return href

st.markdown(file_downloaded(df_selected_team), unsafe_allow_html=True)

# Heatmap
if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')
    df_selected_team.to_csv('output.csv', index=False)
    df = pd.read_csv('output.csv')

    corr = df.select_dtypes(include=[np.number])
    cor = corr.corr()
    mask = np.zeros_like(cor)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cor, mask=mask, vmax=1, square=True, ax=ax)
    st.pyplot(f)