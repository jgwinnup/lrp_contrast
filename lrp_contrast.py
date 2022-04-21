
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import sacrebleu as sb
import pickle

from st_aggrid import AgGrid, GridUpdateMode, GridOptionsBuilder

# show comparison plots for LRP on a per sentence basis
# Averaging may be wonky due to interpolation

# Local config
# lrp_dir = '/tmpssd2/dev/head-story'
# lrp_baseline_file = f'{lrp_dir}/base-lrp-eval/lrp_results'
# lrp_augmented_file = f'{lrp_dir}/yolov5-lrp-eval/lrp_results'
# ref_file = f'{lrp_dir}/test.spm.trim50.pt'

# streamlit cloud config
lrp_dir = '.'
lrp_baseline_file = f'{lrp_dir}/lrp_results/baseline'
lrp_augmented_file = f'{lrp_dir}/lrp_results/yolov5'
ref_file = f'{lrp_dir}/ref/test.spm.trim50.pt'

# Helper for clarity
def desentencepiece(sent):
    return sent.replace(' ', '').replace('▁', ' ').strip()

# For column labels
def tok(sent):
    return sent.split(' ')

# Load and cache reference
@st.cache
def load_ref(ref_file):
    with open(ref_file, 'r') as f:
        lines = [desentencepiece(line.strip()) for line in f]

    return lines


# Load LRP results via pickle
@st.cache
def load_lrp(baseline_file, augmented_file):

    baseline = pickle.load(open(baseline_file, 'rb'))
    augmented = pickle.load(open(augmented_file, 'rb'))

    return baseline, augmented

@st.cache
def get_src_sents(data):
    sents = [desentencepiece(x['src']) for x in data]
    return pd.DataFrame(enumerate(sents), columns=['Id', 'Sent'])


def gen_plot(data, label='inp_lrp'):
    nx, ny = data[label].shape
    fig, ax = plt.subplots()
    plt.yticks(np.arange(0.0, nx, 1.0))
    xtoks = tok(data['dst'])
    xtoks.append('<EOS>')
    ax.set_yticklabels(xtoks)
    plt.xticks(rotation=90)
    plt.xticks(np.arange(0.0, ny, 1.0))
    ytoks = tok(data['src'])
    ytoks.append('<EOS>')
    ax.set_xticklabels(ytoks)
    ax.imshow(data[label], cmap='hot', interpolation='nearest')

    return fig

if __name__ == '__main__':

    # This needs to be first
    st.set_page_config(layout="wide")

    # load LRP results
    lrp_base, lrp_aug = load_lrp(lrp_baseline_file, lrp_augmented_file)
    src_sents = get_src_sents(lrp_base)
    refs = load_ref(ref_file)
    # src_sents = [desentencepiece(x['src']) for x in lrp_base]

    # layout

    st.header("LRP Compare")

    # st.write(f'src_sents is a {type(src_sents)}')

    col1, col2 = st.columns(2)

    sel = None

    with col1:

        st.subheader('Sentence')

        # st.dataframe(src_sents)

        gb = GridOptionsBuilder.from_dataframe(src_sents)
        gb.configure_selection(selection_mode="single", use_checkbox=False)
        go = gb.build()

        res_table = AgGrid(src_sents,
                           gridOptions=go,
                           update_mode=GridUpdateMode.SELECTION_CHANGED,
                           allow_unsafe_jscode=True)

        sel = res_table['selected_rows']

        if len(sel) != 0:
            # st.write(f'Sel is: {sel}')
            id = sel[0]['Id'] # index col of src sents
            # st.write(f'src: {lrp_base[id]}')



    with col2:

        st.subheader('Info')

        if len(sel) != 0:

            ref = refs[id]
            base_hyp = desentencepiece(lrp_base[id]["dst"])
            aug_hyp = desentencepiece(lrp_aug[id]["dst"])

            base_bleu = sb.sentence_bleu(base_hyp, [ref]).score
            aug_bleu = sb.sentence_bleu(aug_hyp, [ref]).score

            st.write(f'{id} ref: {ref}')
            st.write(f'Baseline: ({base_bleu:0.4f}) {base_hyp}')
            st.write(f'Augmented: ({aug_bleu:0.4f}) {aug_hyp}')

            # st.write(f'{id} shape: {lrp_base[id]["inp_lrp"].shape}')
            # st.write(f'{id} src: {tok(lrp_base[id]["src"])}')
            #st.write(f'{id} dst: {tok(lrp_base[id]["dst"])}')

    col3, col4 = st.columns(2)

    with col3:
        if len(sel) != 0:
            st.subheader('Baseline inp_lrp')
            base_fig = gen_plot(lrp_base[id])
            st.pyplot(base_fig)

    with col4:
        if len(sel) != 0:
            st.subheader('Augmented inp_lrp')
            aug_fig = gen_plot(lrp_aug[id])
            st.pyplot(aug_fig)

