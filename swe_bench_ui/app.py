import streamlit as st
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_full_context_diff, run_agentic_inference

st.set_page_config(page_title="SWE-Bench Agentic Coder", layout="wide")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

@st.cache_data
def load_swe_bench_datasets():
    from datasets import load_dataset
    datasets = {}
    for name, split in [
        ("princeton-nlp/SWE-bench_Lite", "dev"),
    ]:
        try:
            ds = load_dataset(name, split=split)
            for item in ds:
                datasets[item['instance_id']] = dict(item)
        except:
            pass
    return datasets

def get_ground_truth(instance_id: str, datasets: dict) -> dict:
    if instance_id in datasets:
        item = datasets[instance_id]
        return {
            'instance_id': item['instance_id'],
            'repo': item['repo'],
            'base_commit': item['base_commit'],
            'problem_statement': item['problem_statement'],
            'hints_text': item.get('hints_text', ''),
            'patch': item['patch'],
            'test_patch': item['test_patch'],
        }
    return None

def load_report(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)

def get_available_reports() -> list:
    if not os.path.exists(RESULTS_DIR):
        return []
    return [f for f in os.listdir(RESULTS_DIR) if f.endswith('.json')]

def display_code_comparison(before: str, after: str, col1_title: str, col2_title: str):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{col1_title}**")
        st.code(before if before else "(empty)", language="python")
    with col2:
        st.markdown(f"**{col2_title}**")
        st.code(after if after else "(empty)", language="python")

def main():
    st.title("SWE-Bench Agentic Coder UI")

    with st.spinner("Loading SWE-bench datasets..."):
        swe_datasets = load_swe_bench_datasets()

    tab1, tab2 = st.tabs(["Report Viewer", "Inference"])

    with tab1:
        st.header("Report Viewer")

        reports = get_available_reports()

        if not reports:
            st.warning("No reports found in results directory")
            return

        selected_report = st.selectbox("Select Report", reports, index=reports.index("agentic_coder_dev.json") if "agentic_coder_dev.json" in reports else 0)

        report_path = os.path.join(RESULTS_DIR, selected_report)
        report = load_report(report_path)

        st.subheader("Report Info")
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.metric("Experiment", report.get("experiment_name", "N/A"))
        with info_col2:
            st.metric("Pipeline", report.get("pipeline_type", "N/A"))
        with info_col3:
            total = len(report.get("instances", []))
            verified = sum(1 for i in report.get("instances", []) if i.get("verified", False))
            st.metric("Success Rate", f"{verified}/{total}")

        instances = report.get("instances", [])
        instance_ids = [i["instance_id"] for i in instances]

        selected_id = st.selectbox("Select Instance", instance_ids)

        selected_instance = next((i for i in instances if i["instance_id"] == selected_id), None)

        if selected_instance:
            st.subheader(f"Instance: {selected_id}")

            status_col1, status_col2 = st.columns(2)
            with status_col1:
                st.metric("Repository", selected_instance.get("repo", "N/A"))
            with status_col2:
                verified = selected_instance.get("verified", False)
                st.metric("Verified", "Yes" if verified else "No")

            attempts = selected_instance.get("attempts", [])
            if attempts:
                latest_attempt = attempts[-1]
                generated_patch = latest_attempt.get("generated_patch", "")

                if latest_attempt.get("error"):
                    st.error(f"Error: {latest_attempt['error']}")

                if generated_patch:
                    st.subheader("Generated Patch")
                    diffs = get_full_context_diff(generated_patch)

                    if diffs:
                        for diff in diffs:
                            st.markdown(f"**File: {diff['file']}**")
                            display_code_comparison(
                                diff['before'],
                                diff['after'],
                                "Before Patch",
                                "After Patch"
                            )
                    else:
                        st.code(generated_patch, language="diff")

            swe_instance = get_ground_truth(selected_id, swe_datasets)
            if swe_instance and swe_instance.get('patch'):
                st.subheader("Ground Truth Patch")
                gt_diffs = get_full_context_diff(swe_instance['patch'])

                if gt_diffs:
                    for diff in gt_diffs:
                        st.markdown(f"**File: {diff['file']}**")
                        display_code_comparison(
                            diff['before'],
                            diff['after'],
                            "Before (Ground Truth)",
                            "After (Ground Truth)"
                        )
                else:
                    st.code(swe_instance['patch'], language="diff")

                with st.expander("Problem Statement"):
                    st.markdown(swe_instance.get('problem_statement', 'N/A'))
            else:
                st.warning(f"Ground truth not found for {selected_id}")

    with tab2:
        st.header("Run Inference")

        st.subheader("Option 1: Load from SWE-Bench")

        swe_instance_id = st.text_input("SWE-Bench Instance ID", placeholder="e.g., sqlfluff__sqlfluff-1625")

        if st.button("Load Instance"):
            if swe_instance_id:
                instance = get_ground_truth(swe_instance_id, swe_datasets)
                if instance:
                    st.session_state['loaded_instance'] = instance
                    st.success(f"Loaded instance: {swe_instance_id}")
                else:
                    st.error("Instance not found in loaded datasets")

        st.subheader("Option 2: Custom Instance")

        with st.expander("Enter Custom Instance Data"):
            custom_repo = st.text_input("Repository", placeholder="owner/repo")
            custom_commit = st.text_input("Base Commit", placeholder="commit hash")
            custom_problem = st.text_area("Problem Statement", placeholder="Describe the bug...")
            custom_hints = st.text_area("Hints (optional)", placeholder="Additional context...")

            if st.button("Create Custom Instance"):
                if custom_repo and custom_commit and custom_problem:
                    st.session_state['loaded_instance'] = {
                        'instance_id': f"custom__{custom_repo.replace('/', '__')}",
                        'repo': custom_repo,
                        'base_commit': custom_commit,
                        'problem_statement': custom_problem,
                        'hints_text': custom_hints,
                        'patch': '',
                        'test_patch': '',
                        'fail_to_pass': '[]',
                        'pass_to_pass': '[]',
                        'environment_setup_commit': custom_commit,
                        'version': ''
                    }
                    st.success("Custom instance created")

        if 'loaded_instance' in st.session_state:
            instance = st.session_state['loaded_instance']
            st.subheader("Loaded Instance")
            st.json({
                'instance_id': instance.get('instance_id'),
                'repo': instance.get('repo'),
                'base_commit': instance.get('base_commit')[:12] + '...' if instance.get('base_commit') else 'N/A'
            })

            with st.expander("Problem Statement"):
                st.markdown(instance.get('problem_statement', 'N/A'))

            if st.button("Run Agentic Inference", type="primary"):
                with st.spinner("Running agentic pipeline..."):
                    result = run_agentic_inference(instance)
                    st.session_state['inference_result'] = result

        if 'inference_result' in st.session_state:
            result = st.session_state['inference_result']
            st.subheader("Inference Result")

            if result['success']:
                st.success("Patch generated successfully")
            else:
                st.error(f"Failed: {result.get('error', 'Unknown error')}")

            if result.get('generated_patch'):
                st.subheader("Generated Patch")
                diffs = get_full_context_diff(result['generated_patch'])

                if diffs:
                    for diff in diffs:
                        st.markdown(f"**File: {diff['file']}**")
                        display_code_comparison(
                            diff['before'],
                            diff['after'],
                            "Before",
                            "After"
                        )
                else:
                    st.code(result['generated_patch'], language="diff")

            if 'loaded_instance' in st.session_state:
                gt_patch = st.session_state['loaded_instance'].get('patch', '')
                if gt_patch:
                    st.subheader("Ground Truth Comparison")
                    gt_diffs = get_full_context_diff(gt_patch)

                    if gt_diffs:
                        for diff in gt_diffs:
                            st.markdown(f"**File: {diff['file']}**")
                            display_code_comparison(
                                diff['before'],
                                diff['after'],
                                "Before (Ground Truth)",
                                "After (Ground Truth)"
                            )

if __name__ == "__main__":
    main()
