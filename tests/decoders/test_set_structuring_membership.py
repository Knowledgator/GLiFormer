"""Entity-to-record attachment modes for the set-structuring decoder."""

import pytest
import torch

from gliformer.tasks.set_structuring.decoder import SetStructuringDecoder


def _claims(probs, valid, active, mode, *, floor=0.0, margin=0.0, threshold=0.5):
    return SetStructuringDecoder._membership_claims(
        torch.tensor(probs),
        torch.tensor(valid, dtype=torch.long),
        None if active is None else torch.tensor(active),
        threshold=threshold,
        mode=mode,
        floor=floor,
        margin=margin,
    )


# Four record slots against three entities. Objectness kept slots 1 and 2, and
# every membership probability on them sits below the 0.5 field threshold --
# the state that returns records whose fields are all null.
PROBS = [
    [0.90, 0.90, 0.90],  # slot 0, not kept by objectness
    [0.30, 0.10, 0.04],  # slot 1, kept
    [0.20, 0.25, 0.02],  # slot 2, kept
    [0.01, 0.01, 0.01],  # slot 3, not kept by objectness
]
ACTIVE = [False, True, True, False]
VALID = [0, 1, 2]


class TestThresholdMode:
    def test_reproduces_the_plain_threshold(self):
        claims = _claims(PROBS, VALID, ACTIVE, "threshold")
        assert torch.equal(claims, torch.tensor(PROBS) > 0.5)

    def test_under_confident_scores_claim_nothing(self):
        # The null case: both live slots lose every field at once.
        claims = _claims(PROBS, VALID, ACTIVE, "threshold")
        assert not claims[1].any()
        assert not claims[2].any()


class TestArgmaxMode:
    def test_every_entity_reaches_exactly_one_record(self):
        claims = _claims(PROBS, VALID, ACTIVE, "argmax")
        assert int(claims.sum()) == len(VALID)

    def test_entities_go_to_their_best_active_record(self):
        claims = _claims(PROBS, VALID, ACTIVE, "argmax")
        assert claims[1, 0] and not claims[2, 0]  # 0.30 > 0.20
        assert claims[2, 1] and not claims[1, 1]  # 0.25 > 0.10
        assert claims[1, 2] and not claims[2, 2]  # 0.04 > 0.02

    def test_records_objectness_rejected_are_never_claimed(self):
        # Slot 0 outscores every live slot but is not a record, so ranking
        # must run over the kept slots only.
        claims = _claims(PROBS, VALID, ACTIVE, "argmax")
        assert not claims[0].any()
        assert not claims[3].any()

    def test_floor_drops_entities_no_record_wants(self):
        claims = _claims(PROBS, VALID, ACTIVE, "argmax", floor=0.1)
        assert not claims[:, 2].any()  # best score 0.04
        assert claims[1, 0] and claims[2, 1]

    def test_margin_shares_a_mention_between_records(self):
        claims = _claims(PROBS, VALID, ACTIVE, "argmax", margin=0.15)
        assert claims[1, 0] and claims[2, 0]  # 0.20 >= 0.30 - 0.15

    def test_margin_leaves_clearly_separated_records_alone(self):
        claims = _claims(PROBS, VALID, ACTIVE, "argmax", margin=0.05)
        assert claims[2, 1] and not claims[1, 1]  # 0.10 < 0.25 - 0.05

    def test_no_entities_yields_no_claims(self):
        assert not _claims(PROBS, [], ACTIVE, "argmax").any()

    def test_no_active_records_yields_no_claims(self):
        assert not _claims(PROBS, VALID, [False] * 4, "argmax").any()

    def test_absent_anchor_mask_ranks_over_every_slot(self):
        claims = _claims(PROBS, VALID, None, "argmax")
        assert claims[0].all()


class TestValidation:
    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match="membership_mode"):
            _claims(PROBS, VALID, ACTIVE, "best")

    def test_floor_outside_unit_interval_rejected(self):
        with pytest.raises(ValueError, match="membership_floor"):
            _claims(PROBS, VALID, ACTIVE, "argmax", floor=1.5)

    def test_negative_margin_rejected(self):
        with pytest.raises(ValueError, match="membership_margin"):
            _claims(PROBS, VALID, ACTIVE, "argmax", margin=-0.1)
