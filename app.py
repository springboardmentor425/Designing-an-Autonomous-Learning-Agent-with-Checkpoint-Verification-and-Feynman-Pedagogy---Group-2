import streamlit as st
# IMPORT CHANGE: We import 'graph_ui' which has the memory attached
from graph import graph_ui as graph

st.title("Web Knowledge Explorer")

query = st.text_input("Enter topic")

if st.button("Search"):
    if query:
        with st.spinner("Searching the web..."):
            # We need a thread_id for the memory saver
            config = {"configurable": {"thread_id": "1"}}
            
            output = graph.invoke(
                {"checkpoint_topic": query},
                config=config
            )
            
            st.success("Search Complete!")
            
            st.caption(f"Source: {output.get('context_source')}")
            
            st.markdown("### Gathered Context")
            st.write(output["gathered_context"])
            
            if output.get("error_message"):
                st.error(f"Error: {output['error_message']}")
    else:
        st.warning("Please enter a topic.")