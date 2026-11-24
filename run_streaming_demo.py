from pathlib import Path
import pandas as pd

from ohe_engine import AuditConfig, OHEInspectionEngine, Streaming10mAuditRunner


def run_chunked_streaming_demo(
    source_csv: str = r"D:\ohe_inspection\ohe_outputs\synthetic_ohe_600k.csv",
    chunk_rows: int = 2000,
    max_batches: int = 12,
    save_chunks_to_inbox: bool = False,
    inbox_dir: str = r"D:\ohe_inspection\stream_inbox",
) -> pd.DataFrame:
    cfg = AuditConfig(
        step_m=10.0,
        bin_release_margin_m=5.0,
        excel_flush_every_bins=10,
    )
    eng = OHEInspectionEngine()
    runner = Streaming10mAuditRunner(eng, config=cfg, csv_filename="ohe_10m_audit_demo.csv")

    inbox_path = Path(inbox_dir)
    if save_chunks_to_inbox:
        inbox_path.mkdir(parents=True, exist_ok=True)

    print(f"Streaming from: {source_csv}")
    print(f"Chunk size: {chunk_rows} rows | Bin size: {cfg.step_m} m | Release margin: +{cfg.bin_release_margin_m} m\n")

    for batch_num, raw_chunk in enumerate(pd.read_csv(source_csv, chunksize=chunk_rows), start=1):
        if max_batches is not None and batch_num > max_batches:
            break

        if save_chunks_to_inbox:
            raw_chunk.to_csv(inbox_path / f"chunk_{batch_num:04d}.csv", index=False)

        c_min = pd.to_numeric(raw_chunk.get("CumDist_m"), errors="coerce").min()
        c_max = pd.to_numeric(raw_chunk.get("CumDist_m"), errors="coerce").max()

        proc_chunk, _ = eng.process_dataframe(raw_chunk)
        finalized = runner.push_batch(proc_chunk)

        open_bins = sorted(runner._bin_buffers.keys())
        done_count = len(runner._finalized_bins)

        print(
            f"[Batch {batch_num:02d}] Rows={len(raw_chunk)} | "
            f"Chainage={c_min:.1f}–{c_max:.1f} m | "
            f"Newly Finalized 10m Bins={len(finalized)} | "
            f"Open Buffered Bins={open_bins} | "
            f"Total Closed Bins={done_count}"
        )

        if not finalized.empty:
            cols = [
                "Chainage_Window_m",
                "Wires_Seen",
                "Incoming_Outgoing_Distance_cm",
                "Outgoing_Stagger_cm",
                "Incoming_Stagger_cm",
                "Pantograph_Contact_Height_m",
                "Takeover_Point",
                "Audit_Status",
            ]
            available_cols = [c for c in cols if c in finalized.columns]
            print(finalized[available_cols].to_string(index=False))
            print("-" * 95)

    print("\n[End of Stream] Flushing remaining buffered 10m bins...")
    tail_df = runner.flush_all(save_excel=True)
    print(f"Total 10m audit rows written to {runner.csv_path}: {len(tail_df)}")
    return tail_df


if __name__ == "__main__":
    run_chunked_streaming_demo(
        chunk_rows=2000,
        max_batches=12,
        save_chunks_to_inbox=False,
    )