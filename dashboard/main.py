import ast
import streamlit as st
from utils.db_operations import retrieve_table_as_df_for_a_date
from utils.read_config import get_tbl_clstr_smmrs
import plotly.express as px

def main():
    st.title("Review Analyser Dashboard")

    # st.subheader("Select Date")
    date = st.date_input("Select Date", value="today", min_value=None, max_value="today")

    # st.write(f"You selected: {date}")
    smmrs = get_tbl_clstr_smmrs()
    smmrs_cols = smmrs["COLUMNS"]

    smmry_data = retrieve_table_as_df_for_a_date(date)
    # smmry_data = smmry_data.drop(smmry_data.columns[0], axis=1)

    all_titles = smmry_data[smmrs_cols['TITLE']].unique().tolist()

    # Multi-select with "Select All" option
    options = ["Select All"] + all_titles
    selected = st.multiselect("Select Titles", options, default=["Select All"])

    # If "Select All" is selected, select all titles
    if "Select All" in selected:
        selected_titles = all_titles
    else:
        selected_titles = selected
    # st.dataframe(smmry_data, use_container_width=True)
    expanded = True
    for title in selected_titles:
        with st.expander(title, expanded=expanded):
            # Filter rows for this title
            title_data = smmry_data[smmry_data[smmrs_cols['TITLE']] == title]
            st.subheader(title)
            st.write(title_data[smmrs_cols['DESCRIPTION']].iloc[0])
            # st.progress(float(title_data[smmrs_cols['VOLUME_PERCENT']]) / 100)
            # st.write(title_data[smmrs_cols['TOP_REVIEWS']].iloc[0])
            st.write("Here are some of the key reviews:")
            reviews_list = ast.literal_eval(title_data[smmrs_cols['TOP_REVIEWS']].iloc[0])
            for review in reviews_list:
                st.write(f"- {review}")
            col1,col2 = st.columns(2)
            with col1:
                # st.metric(label="Volume", value=title_data[smmrs_cols['VOLUME_PERCENT']].iloc[0])
                used = float(title_data[smmrs_cols['VOLUME_PERCENT']].iloc[0])
                remaining = 100 - used

                fig = px.pie(
                    names=[title, "Total"],
                    values=[used, remaining],
                    title="Cluster Volume",
                    hole=0.4
                )
                fig.update_traces(textinfo='label+percent')
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                # st.metric(label="Sentiment", value=title_data[smmrs_cols['SENTIMENT']].iloc[0])
                fig = px.pie(
                    names=["Positive", "Negative", "Neutral"],
                    values=[
                        title_data[smmrs_cols['POSITIVE_COUNT']].iloc[0],
                        title_data[smmrs_cols['NEGATIVE_COUNT']].iloc[0],
                        title_data[smmrs_cols['NEUTRAL_COUNT']].iloc[0]
                    ],
                    title="Sentiment Distribution",
                    color=["Positive", "Negative", "Neutral"],  
                    color_discrete_map = {
                        "Positive": "#4aed83",
                        "Negative": "#f23d3d",
                        "Neutral": "#71a3f5"
                    },
                    hole=0.4
                )
                fig.update_traces(textinfo='label+percent')
                st.plotly_chart(fig, use_container_width=True)
            expanded = False  


if __name__ == "__main__":
    st.set_page_config(page_title="Review Analyser Dashboard", layout="wide")
    main()


