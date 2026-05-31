
def generate_report(
    face_score,
    age_score,
    trajectory_score,
    gnn_score,
    top_zones
):

    final_confidence = (
        0.25 * face_score +
        0.20 * age_score +
        0.40 * trajectory_score +
        0.15 * gnn_score
    )

    print("="*60)
    print("RESCUENET-X REPORT")
    print("="*60)

    print(f"\nFace Match Score: {face_score*100:.2f}%")

    print(
        f"Age-Invariant Confidence: "
        f"{age_score*100:.2f}%"
    )

    print(
        f"Trajectory Confidence: "
        f"{trajectory_score*100:.2f}%"
    )

    print(
        f"GNN Confidence: "
        f"{gnn_score*100:.2f}%"
    )

    print("\nTop Rescue Zones:")

    for rank, (zone, prob) in enumerate(top_zones, 1):

        print(
            f"{rank}. Zone {zone}"
            f" ({prob*100:.2f}%)"
        )

    print("\nFinal Rescue Confidence:")

    print(
        f"{final_confidence*100:.2f}%"
    )

    print("="*60)

    return final_confidence
