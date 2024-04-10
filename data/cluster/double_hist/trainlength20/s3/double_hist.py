import numpy as np
import pandas as pd


def select_closest(keys, queries, predicate):
    scores = [[False for _ in keys] for _ in queries]
    for i, q in enumerate(queries):
        matches = [j for j, k in enumerate(keys) if predicate(q, k)]
        if not (any(matches)):
            scores[i][0] = True
        else:
            j = min(matches, key=lambda j: len(matches) if j == i else abs(i - j))
            scores[i][j] = True
    return scores


def select(keys, queries, predicate):
    return [[predicate(q, k) for k in keys] for q in queries]


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def aggregate_sum(attention, values):
    return [sum([v for a, v in zip(attn, values) if a]) for attn in attention]


def run(tokens):

    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "output/length/rasp/double_hist/trainlength20/s3/double_hist_weights.csv",
        index_col=[0, 1],
        dtype={"feature": str},
    )
    # inputs #####################################################
    token_scores = classifier_weights.loc[[("tokens", str(v)) for v in tokens]]

    positions = list(range(len(tokens)))
    position_scores = classifier_weights.loc[[("positions", str(v)) for v in positions]]

    ones = [1 for _ in range(len(tokens))]
    one_scores = classifier_weights.loc[[("ones", "_") for v in ones]].mul(ones, axis=0)

    # attn_0_0 ####################################################
    def predicate_0_0(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "4"
        elif q_token in {"4", "5"}:
            return k_token == "3"
        elif q_token in {"<s>"}:
            return k_token == "5"

    attn_0_0_pattern = select_closest(tokens, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, positions)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0, 29, 21}:
            return k_position == 4
        elif q_position in {1, 2, 3, 4, 7, 22, 24, 25, 26, 27, 28}:
            return k_position == 5
        elif q_position in {8, 12, 5, 23}:
            return k_position == 6
        elif q_position in {9, 10, 6}:
            return k_position == 2
        elif q_position in {16, 19, 11}:
            return k_position == 10
        elif q_position in {17, 18, 13, 15}:
            return k_position == 14
        elif q_position in {14}:
            return k_position == 8
        elif q_position in {20}:
            return k_position == 23

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, positions)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"5"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_0_2_pattern = select_closest(tokens, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, positions)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_token, k_token):
        if q_token in {"5", "0"}:
            return k_token == "4"
        elif q_token in {"1", "4"}:
            return k_token == "5"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "0"
        elif q_token in {"<s>"}:
            return k_token == "3"

    attn_0_3_pattern = select_closest(tokens, tokens, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, positions)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0, 23}:
            return k_position == 18
        elif q_position in {1, 2, 3, 25}:
            return k_position == 9
        elif q_position in {4}:
            return k_position == 12
        elif q_position in {8, 5, 6}:
            return k_position == 8
        elif q_position in {7}:
            return k_position == 26
        elif q_position in {9}:
            return k_position == 0
        elif q_position in {10}:
            return k_position == 25
        elif q_position in {11}:
            return k_position == 17
        elif q_position in {12}:
            return k_position == 20
        elif q_position in {20, 13}:
            return k_position == 24
        elif q_position in {16, 14}:
            return k_position == 29
        elif q_position in {29, 15}:
            return k_position == 27
        elif q_position in {17}:
            return k_position == 22
        elif q_position in {18}:
            return k_position == 15
        elif q_position in {19}:
            return k_position == 23
        elif q_position in {21}:
            return k_position == 11
        elif q_position in {27, 28, 22}:
            return k_position == 10
        elif q_position in {24, 26}:
            return k_position == 21

    num_attn_0_0_pattern = select(positions, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"5"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == ""

    num_attn_0_1_pattern = select(tokens, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(token, position):
        if token in {"4", "5", "3", "1", "<s>", "0", "2"}:
            return position == 10

    num_attn_0_2_pattern = select(positions, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(token, position):
        if token in {"1", "0", "4"}:
            return position == 16
        elif token in {"2"}:
            return position == 3
        elif token in {"3"}:
            return position == 4
        elif token in {"<s>", "5"}:
            return position == 25

    num_attn_0_3_pattern = select(positions, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_3_output, position):
        key = (attn_0_3_output, position)
        if key in {
            (0, 5),
            (1, 0),
            (1, 1),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 9),
            (1, 10),
            (1, 15),
            (1, 16),
            (1, 19),
            (1, 20),
            (1, 21),
            (1, 22),
            (1, 23),
            (1, 24),
            (1, 25),
            (1, 26),
            (1, 27),
            (1, 28),
            (1, 29),
            (2, 0),
            (2, 1),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 7),
            (2, 9),
            (2, 10),
            (2, 13),
            (2, 14),
            (2, 15),
            (2, 16),
            (2, 17),
            (2, 18),
            (2, 19),
            (2, 20),
            (2, 21),
            (2, 22),
            (2, 23),
            (2, 24),
            (2, 25),
            (2, 26),
            (2, 27),
            (2, 28),
            (2, 29),
            (3, 5),
            (3, 10),
            (4, 0),
            (4, 1),
            (4, 3),
            (4, 4),
            (4, 5),
            (4, 9),
            (4, 10),
            (4, 15),
            (4, 16),
            (4, 19),
            (4, 20),
            (4, 21),
            (4, 22),
            (4, 23),
            (4, 24),
            (4, 25),
            (4, 26),
            (4, 27),
            (4, 28),
            (4, 29),
            (6, 5),
            (7, 5),
            (8, 0),
            (8, 1),
            (8, 3),
            (8, 4),
            (8, 5),
            (8, 7),
            (8, 9),
            (8, 10),
            (8, 13),
            (8, 14),
            (8, 15),
            (8, 16),
            (8, 17),
            (8, 18),
            (8, 19),
            (8, 20),
            (8, 21),
            (8, 22),
            (8, 23),
            (8, 24),
            (8, 25),
            (8, 26),
            (8, 27),
            (8, 28),
            (8, 29),
            (9, 5),
            (10, 0),
            (10, 1),
            (10, 2),
            (10, 3),
            (10, 4),
            (10, 5),
            (10, 6),
            (10, 7),
            (10, 8),
            (10, 9),
            (10, 10),
            (10, 11),
            (10, 12),
            (10, 13),
            (10, 14),
            (10, 15),
            (10, 16),
            (10, 17),
            (10, 18),
            (10, 19),
            (10, 20),
            (10, 21),
            (10, 22),
            (10, 23),
            (10, 24),
            (10, 25),
            (10, 26),
            (10, 27),
            (10, 28),
            (10, 29),
            (12, 5),
            (12, 10),
            (14, 5),
            (16, 0),
            (16, 1),
            (16, 3),
            (16, 4),
            (16, 5),
            (16, 9),
            (16, 10),
            (16, 15),
            (16, 16),
            (16, 19),
            (16, 20),
            (16, 21),
            (16, 22),
            (16, 23),
            (16, 24),
            (16, 25),
            (16, 26),
            (16, 27),
            (16, 28),
            (16, 29),
            (17, 5),
            (17, 10),
            (18, 0),
            (18, 1),
            (18, 3),
            (18, 4),
            (18, 5),
            (18, 7),
            (18, 9),
            (18, 10),
            (18, 13),
            (18, 14),
            (18, 15),
            (18, 16),
            (18, 17),
            (18, 18),
            (18, 19),
            (18, 20),
            (18, 21),
            (18, 22),
            (18, 23),
            (18, 24),
            (18, 25),
            (18, 26),
            (18, 27),
            (18, 28),
            (18, 29),
            (20, 5),
            (20, 10),
            (20, 19),
            (21, 5),
            (21, 10),
            (21, 19),
            (22, 5),
            (22, 10),
            (22, 19),
            (23, 5),
            (23, 10),
            (23, 19),
            (24, 5),
            (24, 10),
            (24, 19),
            (25, 5),
            (25, 10),
            (25, 19),
            (26, 5),
            (26, 10),
            (26, 19),
            (27, 5),
            (27, 10),
            (27, 19),
            (28, 5),
            (28, 10),
            (28, 19),
            (29, 5),
            (29, 10),
            (29, 19),
        }:
            return 10
        elif key in {(11, 17)}:
            return 16
        return 13

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_3_outputs, positions)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_2_output, token):
        key = (attn_0_2_output, token)
        if key in {
            (0, "0"),
            (2, "0"),
            (3, "0"),
            (4, "0"),
            (5, "0"),
            (6, "0"),
            (8, "0"),
            (9, "0"),
            (11, "0"),
            (12, "0"),
            (13, "0"),
            (14, "0"),
            (16, "0"),
            (18, "0"),
            (20, "0"),
            (21, "0"),
            (22, "0"),
            (23, "0"),
            (24, "0"),
            (25, "0"),
            (26, "0"),
            (27, "0"),
            (28, "0"),
            (29, "0"),
        }:
            return 15
        elif key in {(18, "2"), (18, "4"), (18, "<s>")}:
            return 6
        return 26

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_2_outputs, tokens)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_1_output, num_attn_0_2_output):
        key = (num_attn_0_1_output, num_attn_0_2_output)
        if key in {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 0), (1, 1)}:
            return 21
        return 10

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_1_output, num_attn_0_2_output):
        key = (num_attn_0_1_output, num_attn_0_2_output)
        if key in {
            (0, 21),
            (0, 22),
            (0, 23),
            (0, 24),
            (0, 25),
            (0, 26),
            (0, 27),
            (0, 28),
            (0, 29),
            (1, 23),
            (1, 24),
            (1, 25),
            (1, 26),
            (1, 27),
            (1, 28),
            (1, 29),
            (2, 26),
            (2, 27),
            (2, 28),
            (2, 29),
            (3, 28),
            (3, 29),
            (27, 0),
            (28, 0),
            (28, 1),
            (29, 0),
            (29, 1),
        }:
            return 24
        elif key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (1, 0),
            (1, 1),
            (1, 2),
        }:
            return 19
        elif key in {(2, 0)}:
            return 14
        return 11

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(token, num_mlp_0_0_output):
        if token in {"1", "5", "0", "4"}:
            return num_mlp_0_0_output == 10
        elif token in {"2", "3"}:
            return num_mlp_0_0_output == 11
        elif token in {"<s>"}:
            return num_mlp_0_0_output == 20

    attn_1_0_pattern = select_closest(num_mlp_0_0_outputs, tokens, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_2_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(token, position):
        if token in {"0"}:
            return position == 4
        elif token in {"1"}:
            return position == 6
        elif token in {"2"}:
            return position == 12
        elif token in {"3"}:
            return position == 8
        elif token in {"4"}:
            return position == 1
        elif token in {"5"}:
            return position == 14
        elif token in {"<s>"}:
            return position == 3

    attn_1_1_pattern = select_closest(positions, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_0_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_1_output, position):
        if attn_0_1_output in {0, 20, 22, 23, 24, 25, 26, 29}:
            return position == 4
        elif attn_0_1_output in {1, 3, 28}:
            return position == 23
        elif attn_0_1_output in {2, 4, 6}:
            return position == 7
        elif attn_0_1_output in {10, 5}:
            return position == 6
        elif attn_0_1_output in {11, 7}:
            return position == 1
        elif attn_0_1_output in {8}:
            return position == 3
        elif attn_0_1_output in {9}:
            return position == 17
        elif attn_0_1_output in {12, 15}:
            return position == 8
        elif attn_0_1_output in {13}:
            return position == 15
        elif attn_0_1_output in {14}:
            return position == 16
        elif attn_0_1_output in {16}:
            return position == 11
        elif attn_0_1_output in {17}:
            return position == 14
        elif attn_0_1_output in {18}:
            return position == 12
        elif attn_0_1_output in {19}:
            return position == 13
        elif attn_0_1_output in {21}:
            return position == 19
        elif attn_0_1_output in {27}:
            return position == 24

    attn_1_2_pattern = select_closest(positions, attn_0_1_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, positions)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_position, k_position):
        if q_position in {0, 1, 2, 3, 4, 8, 16, 17, 26, 27, 29}:
            return k_position == 5
        elif q_position in {5, 6, 10, 13, 19}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 2
        elif q_position in {9, 12}:
            return k_position == 11
        elif q_position in {18, 11}:
            return k_position == 6
        elif q_position in {14, 15}:
            return k_position == 8
        elif q_position in {20}:
            return k_position == 18
        elif q_position in {21}:
            return k_position == 25
        elif q_position in {24, 22}:
            return k_position == 4
        elif q_position in {23}:
            return k_position == 28
        elif q_position in {25}:
            return k_position == 15
        elif q_position in {28}:
            return k_position == 20

    attn_1_3_pattern = select_closest(positions, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_1_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(num_mlp_0_0_output, position):
        if num_mlp_0_0_output in {0, 3, 9, 15, 16, 19, 23, 26}:
            return position == 24
        elif num_mlp_0_0_output in {1, 2, 8, 13, 22}:
            return position == 20
        elif num_mlp_0_0_output in {4, 7}:
            return position == 26
        elif num_mlp_0_0_output in {29, 12, 5}:
            return position == 25
        elif num_mlp_0_0_output in {6}:
            return position == 23
        elif num_mlp_0_0_output in {10, 17, 18, 25, 27}:
            return position == 29
        elif num_mlp_0_0_output in {24, 11, 28}:
            return position == 27
        elif num_mlp_0_0_output in {20, 14}:
            return position == 28
        elif num_mlp_0_0_output in {21}:
            return position == 6

    num_attn_1_0_pattern = select(positions, num_mlp_0_0_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, ones)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(num_mlp_0_0_output, position):
        if num_mlp_0_0_output in {0, 1, 2, 3, 22, 25, 28}:
            return position == 0
        elif num_mlp_0_0_output in {4}:
            return position == 4
        elif num_mlp_0_0_output in {5}:
            return position == 5
        elif num_mlp_0_0_output in {6, 14}:
            return position == 22
        elif num_mlp_0_0_output in {8, 20, 23, 7}:
            return position == 25
        elif num_mlp_0_0_output in {9, 18}:
            return position == 27
        elif num_mlp_0_0_output in {10, 13}:
            return position == 23
        elif num_mlp_0_0_output in {16, 26, 11}:
            return position == 26
        elif num_mlp_0_0_output in {12}:
            return position == 24
        elif num_mlp_0_0_output in {15}:
            return position == 21
        elif num_mlp_0_0_output in {17, 19, 29}:
            return position == 28
        elif num_mlp_0_0_output in {21}:
            return position == 2
        elif num_mlp_0_0_output in {24}:
            return position == 20
        elif num_mlp_0_0_output in {27}:
            return position == 29

    num_attn_1_1_pattern = select(positions, num_mlp_0_0_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_1_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(q_num_mlp_0_0_output, k_num_mlp_0_0_output):
        if q_num_mlp_0_0_output in {0, 16, 26}:
            return k_num_mlp_0_0_output == 20
        elif q_num_mlp_0_0_output in {1, 5, 15}:
            return k_num_mlp_0_0_output == 29
        elif q_num_mlp_0_0_output in {2, 3, 4, 6}:
            return k_num_mlp_0_0_output == 25
        elif q_num_mlp_0_0_output in {7, 8, 10, 13, 18, 19}:
            return k_num_mlp_0_0_output == 28
        elif q_num_mlp_0_0_output in {9}:
            return k_num_mlp_0_0_output == 26
        elif q_num_mlp_0_0_output in {11}:
            return k_num_mlp_0_0_output == 4
        elif q_num_mlp_0_0_output in {12, 14, 22}:
            return k_num_mlp_0_0_output == 24
        elif q_num_mlp_0_0_output in {17}:
            return k_num_mlp_0_0_output == 22
        elif q_num_mlp_0_0_output in {25, 20}:
            return k_num_mlp_0_0_output == 27
        elif q_num_mlp_0_0_output in {21}:
            return k_num_mlp_0_0_output == 21
        elif q_num_mlp_0_0_output in {27, 23}:
            return k_num_mlp_0_0_output == 15
        elif q_num_mlp_0_0_output in {24}:
            return k_num_mlp_0_0_output == 23
        elif q_num_mlp_0_0_output in {28, 29}:
            return k_num_mlp_0_0_output == 19

    num_attn_1_2_pattern = select(
        num_mlp_0_0_outputs, num_mlp_0_0_outputs, num_predicate_1_2
    )
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, ones)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_1_output, position):
        if attn_0_1_output in {0, 4, 20, 22, 26}:
            return position == 8
        elif attn_0_1_output in {1}:
            return position == 9
        elif attn_0_1_output in {2, 3, 6, 7, 25, 27}:
            return position == 18
        elif attn_0_1_output in {5}:
            return position == 6
        elif attn_0_1_output in {8}:
            return position == 20
        elif attn_0_1_output in {24, 9}:
            return position == 17
        elif attn_0_1_output in {10}:
            return position == 22
        elif attn_0_1_output in {16, 19, 11}:
            return position == 21
        elif attn_0_1_output in {12, 29}:
            return position == 19
        elif attn_0_1_output in {18, 13, 15}:
            return position == 29
        elif attn_0_1_output in {14}:
            return position == 23
        elif attn_0_1_output in {17}:
            return position == 25
        elif attn_0_1_output in {21}:
            return position == 14
        elif attn_0_1_output in {23}:
            return position == 0
        elif attn_0_1_output in {28}:
            return position == 28

    num_attn_1_3_pattern = select(positions, attn_0_1_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_0_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_3_output, attn_1_2_output):
        key = (attn_0_3_output, attn_1_2_output)
        return 0

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_1_2_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_0_2_output, attn_1_1_output):
        key = (attn_0_2_output, attn_1_1_output)
        if key in {
            (0, 3),
            (0, 9),
            (0, 11),
            (0, 16),
            (2, 9),
            (3, 9),
            (4, 2),
            (4, 3),
            (4, 8),
            (4, 9),
            (4, 11),
            (4, 12),
            (4, 15),
            (4, 16),
            (4, 18),
            (4, 19),
            (4, 20),
            (4, 21),
            (4, 23),
            (4, 25),
            (4, 28),
            (4, 29),
            (5, 9),
            (6, 9),
            (7, 9),
            (8, 9),
            (9, 9),
            (10, 2),
            (10, 3),
            (10, 8),
            (10, 9),
            (10, 11),
            (10, 12),
            (10, 15),
            (10, 16),
            (10, 18),
            (10, 19),
            (10, 20),
            (10, 21),
            (10, 23),
            (10, 25),
            (10, 28),
            (11, 9),
            (12, 3),
            (12, 9),
            (12, 11),
            (12, 16),
            (13, 9),
            (14, 2),
            (14, 3),
            (14, 8),
            (14, 9),
            (14, 11),
            (14, 12),
            (14, 15),
            (14, 16),
            (14, 18),
            (14, 19),
            (14, 20),
            (14, 21),
            (14, 23),
            (14, 25),
            (14, 28),
            (15, 9),
            (16, 2),
            (16, 3),
            (16, 8),
            (16, 9),
            (16, 11),
            (16, 12),
            (16, 15),
            (16, 16),
            (16, 18),
            (16, 19),
            (16, 20),
            (16, 21),
            (16, 23),
            (16, 25),
            (16, 28),
            (17, 2),
            (17, 3),
            (17, 8),
            (17, 9),
            (17, 11),
            (17, 12),
            (17, 15),
            (17, 16),
            (17, 18),
            (17, 19),
            (17, 20),
            (17, 21),
            (17, 22),
            (17, 23),
            (17, 25),
            (17, 26),
            (17, 27),
            (17, 28),
            (17, 29),
            (19, 3),
            (19, 9),
            (19, 11),
            (19, 16),
            (20, 3),
            (20, 9),
            (20, 11),
            (20, 16),
            (21, 9),
            (22, 9),
            (23, 9),
            (24, 3),
            (24, 9),
            (24, 11),
            (24, 16),
            (25, 9),
            (26, 9),
            (27, 9),
            (28, 3),
            (28, 9),
            (28, 11),
            (28, 16),
            (29, 9),
        }:
            return 8
        return 0

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_1_1_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_0_output, num_attn_1_3_output):
        key = (num_attn_1_0_output, num_attn_1_3_output)
        return 24

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_0_output, num_attn_1_3_output):
        key = (num_attn_0_0_output, num_attn_1_3_output)
        return 2

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_position, k_position):
        if q_position in {0, 20, 21, 22, 27, 28, 29}:
            return k_position == 6
        elif q_position in {1, 3, 5, 6, 7, 8}:
            return k_position == 12
        elif q_position in {2, 11}:
            return k_position == 4
        elif q_position in {4}:
            return k_position == 7
        elif q_position in {9, 19}:
            return k_position == 3
        elif q_position in {16, 10}:
            return k_position == 11
        elif q_position in {12}:
            return k_position == 2
        elif q_position in {18, 13, 15}:
            return k_position == 19
        elif q_position in {14, 23}:
            return k_position == 18
        elif q_position in {17}:
            return k_position == 5
        elif q_position in {24}:
            return k_position == 23
        elif q_position in {25}:
            return k_position == 26
        elif q_position in {26}:
            return k_position == 24

    attn_2_0_pattern = select_closest(positions, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_3_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_0_1_output, num_mlp_0_0_output):
        if attn_0_1_output in {0, 10, 29, 14}:
            return num_mlp_0_0_output == 11
        elif attn_0_1_output in {1, 2, 3, 4, 5, 6, 9, 12, 19}:
            return num_mlp_0_0_output == 23
        elif attn_0_1_output in {23, 7}:
            return num_mlp_0_0_output == 18
        elif attn_0_1_output in {8}:
            return num_mlp_0_0_output == 10
        elif attn_0_1_output in {11}:
            return num_mlp_0_0_output == 19
        elif attn_0_1_output in {13}:
            return num_mlp_0_0_output == 17
        elif attn_0_1_output in {15}:
            return num_mlp_0_0_output == 1
        elif attn_0_1_output in {16}:
            return num_mlp_0_0_output == 20
        elif attn_0_1_output in {17}:
            return num_mlp_0_0_output == 29
        elif attn_0_1_output in {18}:
            return num_mlp_0_0_output == 26
        elif attn_0_1_output in {20}:
            return num_mlp_0_0_output == 6
        elif attn_0_1_output in {21}:
            return num_mlp_0_0_output == 28
        elif attn_0_1_output in {24, 22}:
            return num_mlp_0_0_output == 15
        elif attn_0_1_output in {25}:
            return num_mlp_0_0_output == 3
        elif attn_0_1_output in {26}:
            return num_mlp_0_0_output == 27
        elif attn_0_1_output in {27}:
            return num_mlp_0_0_output == 22
        elif attn_0_1_output in {28}:
            return num_mlp_0_0_output == 2

    attn_2_1_pattern = select_closest(
        num_mlp_0_0_outputs, attn_0_1_outputs, predicate_2_1
    )
    attn_2_1_outputs = aggregate(attn_2_1_pattern, num_mlp_0_0_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_token, k_token):
        if q_token in {"1", "0"}:
            return k_token == "4"
        elif q_token in {"2"}:
            return k_token == "5"
        elif q_token in {"5", "3"}:
            return k_token == "2"
        elif q_token in {"4"}:
            return k_token == "0"
        elif q_token in {"<s>"}:
            return k_token == "1"

    attn_2_2_pattern = select_closest(tokens, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_0_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_0_output, num_mlp_0_0_output):
        if attn_0_0_output in {0, 2, 14}:
            return num_mlp_0_0_output == 20
        elif attn_0_0_output in {1, 3, 5}:
            return num_mlp_0_0_output == 23
        elif attn_0_0_output in {4, 21}:
            return num_mlp_0_0_output == 28
        elif attn_0_0_output in {17, 28, 29, 6}:
            return num_mlp_0_0_output == 17
        elif attn_0_0_output in {19, 7}:
            return num_mlp_0_0_output == 11
        elif attn_0_0_output in {8, 20}:
            return num_mlp_0_0_output == 27
        elif attn_0_0_output in {9, 10}:
            return num_mlp_0_0_output == 12
        elif attn_0_0_output in {25, 11}:
            return num_mlp_0_0_output == 2
        elif attn_0_0_output in {12, 15}:
            return num_mlp_0_0_output == 7
        elif attn_0_0_output in {13}:
            return num_mlp_0_0_output == 19
        elif attn_0_0_output in {16}:
            return num_mlp_0_0_output == 26
        elif attn_0_0_output in {18, 26}:
            return num_mlp_0_0_output == 25
        elif attn_0_0_output in {27, 22}:
            return num_mlp_0_0_output == 24
        elif attn_0_0_output in {24, 23}:
            return num_mlp_0_0_output == 15

    attn_2_3_pattern = select_closest(
        num_mlp_0_0_outputs, attn_0_0_outputs, predicate_2_3
    )
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_2_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_0_output, attn_0_0_output):
        if attn_1_0_output in {0, 2}:
            return attn_0_0_output == 1
        elif attn_1_0_output in {1}:
            return attn_0_0_output == 4
        elif attn_1_0_output in {17, 3, 5}:
            return attn_0_0_output == 2
        elif attn_1_0_output in {8, 9, 4}:
            return attn_0_0_output == 3
        elif attn_1_0_output in {21, 6}:
            return attn_0_0_output == 5
        elif attn_1_0_output in {7}:
            return attn_0_0_output == 9
        elif attn_1_0_output in {16, 10, 12}:
            return attn_0_0_output == 24
        elif attn_1_0_output in {25, 11}:
            return attn_0_0_output == 25
        elif attn_1_0_output in {24, 13}:
            return attn_0_0_output == 22
        elif attn_1_0_output in {14}:
            return attn_0_0_output == 27
        elif attn_1_0_output in {15}:
            return attn_0_0_output == 11
        elif attn_1_0_output in {18, 20, 23}:
            return attn_0_0_output == 29
        elif attn_1_0_output in {27, 19}:
            return attn_0_0_output == 28
        elif attn_1_0_output in {26, 22}:
            return attn_0_0_output == 19
        elif attn_1_0_output in {28}:
            return attn_0_0_output == 23
        elif attn_1_0_output in {29}:
            return attn_0_0_output == 20

    num_attn_2_0_pattern = select(attn_0_0_outputs, attn_1_0_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_2_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_0_output, position):
        if attn_1_0_output in {0, 1, 24}:
            return position == 1
        elif attn_1_0_output in {2, 29}:
            return position == 2
        elif attn_1_0_output in {3, 23}:
            return position == 3
        elif attn_1_0_output in {4, 21}:
            return position == 4
        elif attn_1_0_output in {27, 28, 5}:
            return position == 5
        elif attn_1_0_output in {6, 20, 22, 25, 26}:
            return position == 6
        elif attn_1_0_output in {7}:
            return position == 7
        elif attn_1_0_output in {8}:
            return position == 8
        elif attn_1_0_output in {9}:
            return position == 9
        elif attn_1_0_output in {10}:
            return position == 10
        elif attn_1_0_output in {11}:
            return position == 11
        elif attn_1_0_output in {12}:
            return position == 12
        elif attn_1_0_output in {13}:
            return position == 13
        elif attn_1_0_output in {14}:
            return position == 14
        elif attn_1_0_output in {17, 15}:
            return position == 15
        elif attn_1_0_output in {16, 19}:
            return position == 16
        elif attn_1_0_output in {18}:
            return position == 18

    num_attn_2_1_pattern = select(positions, attn_1_0_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_1_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_0_output, attn_1_1_output):
        if attn_1_0_output in {0, 24, 4}:
            return attn_1_1_output == 11
        elif attn_1_0_output in {1, 26, 5, 23}:
            return attn_1_1_output == 15
        elif attn_1_0_output in {2, 13}:
            return attn_1_1_output == 9
        elif attn_1_0_output in {27, 3}:
            return attn_1_1_output == 20
        elif attn_1_0_output in {6, 7}:
            return attn_1_1_output == 16
        elif attn_1_0_output in {8}:
            return attn_1_1_output == 12
        elif attn_1_0_output in {9, 14}:
            return attn_1_1_output == 10
        elif attn_1_0_output in {10, 21}:
            return attn_1_1_output == 8
        elif attn_1_0_output in {11}:
            return attn_1_1_output == 14
        elif attn_1_0_output in {17, 18, 12}:
            return attn_1_1_output == 7
        elif attn_1_0_output in {15}:
            return attn_1_1_output == 4
        elif attn_1_0_output in {16, 20}:
            return attn_1_1_output == 17
        elif attn_1_0_output in {19}:
            return attn_1_1_output == 19
        elif attn_1_0_output in {22}:
            return attn_1_1_output == 13
        elif attn_1_0_output in {25}:
            return attn_1_1_output == 18
        elif attn_1_0_output in {28}:
            return attn_1_1_output == 24
        elif attn_1_0_output in {29}:
            return attn_1_1_output == 27

    num_attn_2_2_pattern = select(attn_1_1_outputs, attn_1_0_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_2_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(num_mlp_0_0_output, position):
        if num_mlp_0_0_output in {0, 14, 22}:
            return position == 26
        elif num_mlp_0_0_output in {1}:
            return position == 24
        elif num_mlp_0_0_output in {27, 2, 3}:
            return position == 28
        elif num_mlp_0_0_output in {4}:
            return position == 22
        elif num_mlp_0_0_output in {8, 12, 5, 15}:
            return position == 27
        elif num_mlp_0_0_output in {16, 9, 6}:
            return position == 23
        elif num_mlp_0_0_output in {10, 13, 7}:
            return position == 29
        elif num_mlp_0_0_output in {11}:
            return position == 25
        elif num_mlp_0_0_output in {17}:
            return position == 20
        elif num_mlp_0_0_output in {18, 19, 20, 23, 24, 25, 26, 28}:
            return position == 21
        elif num_mlp_0_0_output in {21}:
            return position == 8
        elif num_mlp_0_0_output in {29}:
            return position == 0

    num_attn_2_3_pattern = select(positions, num_mlp_0_0_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, ones)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(num_mlp_0_0_output, num_mlp_1_1_output):
        key = (num_mlp_0_0_output, num_mlp_1_1_output)
        if key in {
            (16, 0),
            (16, 3),
            (16, 4),
            (16, 5),
            (16, 6),
            (16, 7),
            (16, 10),
            (16, 11),
            (16, 12),
            (16, 14),
            (16, 16),
            (16, 17),
            (16, 18),
            (16, 20),
            (16, 21),
            (16, 22),
            (16, 23),
            (16, 27),
            (16, 28),
            (16, 29),
            (22, 6),
            (22, 16),
            (22, 20),
            (26, 6),
            (26, 20),
            (28, 16),
        }:
            return 23
        return 4

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(num_mlp_0_0_outputs, num_mlp_1_1_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_1_1_output, attn_2_3_output):
        key = (attn_1_1_output, attn_2_3_output)
        return 7

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_2_3_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_2_output, num_attn_2_3_output):
        key = (num_attn_1_2_output, num_attn_2_3_output)
        return 5

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_2_3_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_3_output, num_attn_2_1_output):
        key = (num_attn_2_3_output, num_attn_2_1_output)
        return 11

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_3_outputs, num_attn_2_1_outputs)
    ]
    num_mlp_2_1_output_scores = classifier_weights.loc[
        [("num_mlp_2_1_outputs", str(v)) for v in num_mlp_2_1_outputs]
    ]

    feature_logits = pd.concat(
        [
            df.reset_index()
            for df in [
                token_scores,
                position_scores,
                attn_0_0_output_scores,
                attn_0_1_output_scores,
                attn_0_2_output_scores,
                attn_0_3_output_scores,
                mlp_0_0_output_scores,
                mlp_0_1_output_scores,
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                mlp_1_0_output_scores,
                mlp_1_1_output_scores,
                num_mlp_1_0_output_scores,
                num_mlp_1_1_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                attn_2_2_output_scores,
                attn_2_3_output_scores,
                mlp_2_0_output_scores,
                mlp_2_1_output_scores,
                num_mlp_2_0_output_scores,
                num_mlp_2_1_output_scores,
                one_scores,
                num_attn_0_0_output_scores,
                num_attn_0_1_output_scores,
                num_attn_0_2_output_scores,
                num_attn_0_3_output_scores,
                num_attn_1_0_output_scores,
                num_attn_1_1_output_scores,
                num_attn_1_2_output_scores,
                num_attn_1_3_output_scores,
                num_attn_2_0_output_scores,
                num_attn_2_1_output_scores,
                num_attn_2_2_output_scores,
                num_attn_2_3_output_scores,
            ]
        ]
    )
    logits = feature_logits.groupby(level=0).sum(numeric_only=True).to_numpy()
    classes = classifier_weights.columns.to_numpy()
    predictions = classes[logits.argmax(-1)]
    if tokens[0] == "<s>":
        predictions[0] = "<s>"
    if tokens[-1] == "</s>":
        predictions[-1] = "</s>"
    return predictions.tolist()


print(run(["<s>", "0", "1", "3", "0", "0", "0", "5", "5", "3", "2", "3"]))
