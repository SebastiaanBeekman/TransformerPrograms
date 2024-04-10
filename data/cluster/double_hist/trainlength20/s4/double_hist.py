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
        "output/length/rasp/double_hist/trainlength20/s4/double_hist_weights.csv",
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
    def predicate_0_0(q_position, k_position):
        if q_position in {0, 2, 4, 20, 24, 25, 28}:
            return k_position == 4
        elif q_position in {1}:
            return k_position == 5
        elif q_position in {19, 3}:
            return k_position == 9
        elif q_position in {5, 6}:
            return k_position == 6
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {8, 9, 10, 11, 12, 14}:
            return k_position == 1
        elif q_position in {27, 13}:
            return k_position == 11
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {16, 17}:
            return k_position == 15
        elif q_position in {18, 21}:
            return k_position == 13
        elif q_position in {22}:
            return k_position == 24
        elif q_position in {23}:
            return k_position == 0
        elif q_position in {26}:
            return k_position == 10
        elif q_position in {29}:
            return k_position == 12

    attn_0_0_pattern = select_closest(positions, positions, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, positions)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "2"
        elif q_token in {"1"}:
            return k_token == "5"
        elif q_token in {"2"}:
            return k_token == "1"
        elif q_token in {"3"}:
            return k_token == "4"
        elif q_token in {"4"}:
            return k_token == "3"
        elif q_token in {"5"}:
            return k_token == "0"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_0_1_pattern = select_closest(tokens, tokens, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, positions)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "1"
        elif q_token in {"1"}:
            return k_token == "3"
        elif q_token in {"2"}:
            return k_token == "4"
        elif q_token in {"3", "5"}:
            return k_token == "2"
        elif q_token in {"4"}:
            return k_token == "0"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_0_2_pattern = select_closest(tokens, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, positions)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 4
        elif q_position in {1, 3, 20, 21, 22, 23, 24, 25, 28, 29}:
            return k_position == 5
        elif q_position in {2}:
            return k_position == 15
        elif q_position in {26, 27, 4, 5}:
            return k_position == 6
        elif q_position in {19, 6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 8
        elif q_position in {8, 9, 10, 16}:
            return k_position == 1
        elif q_position in {11, 12}:
            return k_position == 3
        elif q_position in {13}:
            return k_position == 11
        elif q_position in {17, 14}:
            return k_position == 10
        elif q_position in {15}:
            return k_position == 2
        elif q_position in {18}:
            return k_position == 12

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, positions)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_token, k_token):
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

    num_attn_0_0_pattern = select(tokens, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0, 23}:
            return k_position == 21
        elif q_position in {1, 4, 11, 14, 18}:
            return k_position == 3
        elif q_position in {8, 16, 2}:
            return k_position == 1
        elif q_position in {3}:
            return k_position == 2
        elif q_position in {5}:
            return k_position == 4
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {7}:
            return k_position == 7
        elif q_position in {9, 20}:
            return k_position == 9
        elif q_position in {24, 17, 10, 27}:
            return k_position == 8
        elif q_position in {12}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 6
        elif q_position in {15}:
            return k_position == 10
        elif q_position in {19}:
            return k_position == 16
        elif q_position in {25, 29, 21}:
            return k_position == 28
        elif q_position in {22}:
            return k_position == 29
        elif q_position in {26}:
            return k_position == 23
        elif q_position in {28}:
            return k_position == 19

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 2
        elif q_position in {1, 3, 5, 7, 9, 10}:
            return k_position == 18
        elif q_position in {2, 22, 14, 15}:
            return k_position == 16
        elif q_position in {11, 4}:
            return k_position == 17
        elif q_position in {6}:
            return k_position == 13
        elif q_position in {8, 12}:
            return k_position == 20
        elif q_position in {13}:
            return k_position == 23
        elif q_position in {16}:
            return k_position == 7
        elif q_position in {17, 26, 25}:
            return k_position == 19
        elif q_position in {18}:
            return k_position == 15
        elif q_position in {27, 19, 29}:
            return k_position == 28
        elif q_position in {20}:
            return k_position == 27
        elif q_position in {21}:
            return k_position == 25
        elif q_position in {23}:
            return k_position == 21
        elif q_position in {24, 28}:
            return k_position == 26

    num_attn_0_2_pattern = select(positions, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_token, k_token):
        if q_token in {"1", "2", "3", "5", "4", "0", "<s>"}:
            return k_token == ""

    num_attn_0_3_pattern = select(tokens, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_1_output, attn_0_3_output):
        key = (attn_0_1_output, attn_0_3_output)
        return 13

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_1_outputs, attn_0_3_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_3_output, attn_0_0_output):
        key = (attn_0_3_output, attn_0_0_output)
        if key in {
            (14, 2),
            (14, 7),
            (14, 19),
            (15, 0),
            (15, 1),
            (15, 2),
            (15, 3),
            (15, 4),
            (15, 5),
            (15, 6),
            (15, 7),
            (15, 8),
            (15, 9),
            (15, 10),
            (15, 11),
            (15, 12),
            (15, 13),
            (15, 14),
            (15, 15),
            (15, 16),
            (15, 17),
            (15, 18),
            (15, 19),
            (15, 20),
            (15, 21),
            (15, 22),
            (15, 23),
            (15, 24),
            (15, 25),
            (15, 26),
            (15, 27),
            (15, 28),
            (15, 29),
        }:
            return 5
        return 3

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_3_outputs, attn_0_0_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_2_output, num_attn_0_3_output):
        key = (num_attn_0_2_output, num_attn_0_3_output)
        return 29

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_3_output, num_attn_0_2_output):
        key = (num_attn_0_3_output, num_attn_0_2_output)
        return 9

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_3_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_0_output, position):
        if attn_0_0_output in {0, 4}:
            return position == 5
        elif attn_0_0_output in {1, 9, 10, 11, 29}:
            return position == 10
        elif attn_0_0_output in {2, 8, 13, 15, 18}:
            return position == 9
        elif attn_0_0_output in {17, 3, 5, 7}:
            return position == 8
        elif attn_0_0_output in {6}:
            return position == 7
        elif attn_0_0_output in {12}:
            return position == 1
        elif attn_0_0_output in {26, 28, 14, 23}:
            return position == 13
        elif attn_0_0_output in {16, 19}:
            return position == 14
        elif attn_0_0_output in {20}:
            return position == 12
        elif attn_0_0_output in {21}:
            return position == 21
        elif attn_0_0_output in {22}:
            return position == 25
        elif attn_0_0_output in {24, 27}:
            return position == 4
        elif attn_0_0_output in {25}:
            return position == 15

    attn_1_0_pattern = select_closest(positions, attn_0_0_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_0_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(attn_0_0_output, position):
        if attn_0_0_output in {0, 4, 5, 6, 14}:
            return position == 7
        elif attn_0_0_output in {1, 25}:
            return position == 13
        elif attn_0_0_output in {17, 2}:
            return position == 16
        elif attn_0_0_output in {11, 3}:
            return position == 10
        elif attn_0_0_output in {7}:
            return position == 9
        elif attn_0_0_output in {8}:
            return position == 11
        elif attn_0_0_output in {9}:
            return position == 12
        elif attn_0_0_output in {24, 10, 22, 15}:
            return position == 8
        elif attn_0_0_output in {19, 12, 13}:
            return position == 15
        elif attn_0_0_output in {16}:
            return position == 14
        elif attn_0_0_output in {18, 29}:
            return position == 17
        elif attn_0_0_output in {26, 28, 20, 21}:
            return position == 5
        elif attn_0_0_output in {23}:
            return position == 28
        elif attn_0_0_output in {27}:
            return position == 6

    attn_1_1_pattern = select_closest(positions, attn_0_0_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_3_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_token, k_token):
        if q_token in {"3", "4", "0"}:
            return k_token == ""
        elif q_token in {"1", "5"}:
            return k_token == "4"
        elif q_token in {"2"}:
            return k_token == "1"
        elif q_token in {"<s>"}:
            return k_token == "5"

    attn_1_2_pattern = select_closest(tokens, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, num_mlp_0_1_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_token, k_token):
        if q_token in {"5", "0", "2"}:
            return k_token == "4"
        elif q_token in {"1"}:
            return k_token == "0"
        elif q_token in {"3", "4"}:
            return k_token == "1"
        elif q_token in {"<s>"}:
            return k_token == "5"

    attn_1_3_pattern = select_closest(tokens, tokens, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_1_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(q_token, k_token):
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
            return k_token == "<s>"

    num_attn_1_0_pattern = select(tokens, tokens, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, ones)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_2_output, position):
        if attn_0_2_output in {0}:
            return position == 6
        elif attn_0_2_output in {24, 1}:
            return position == 26
        elif attn_0_2_output in {2, 11, 4}:
            return position == 17
        elif attn_0_2_output in {3, 5, 6, 8, 12, 26}:
            return position == 18
        elif attn_0_2_output in {27, 7}:
            return position == 14
        elif attn_0_2_output in {9, 19}:
            return position == 29
        elif attn_0_2_output in {10}:
            return position == 23
        elif attn_0_2_output in {13}:
            return position == 28
        elif attn_0_2_output in {14}:
            return position == 22
        elif attn_0_2_output in {25, 23, 15}:
            return position == 21
        elif attn_0_2_output in {16, 17, 18}:
            return position == 27
        elif attn_0_2_output in {29, 20, 28}:
            return position == 12
        elif attn_0_2_output in {21}:
            return position == 11
        elif attn_0_2_output in {22}:
            return position == 24

    num_attn_1_1_pattern = select(positions, attn_0_2_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_2_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_0_output, position):
        if attn_0_0_output in {0, 1, 3, 4, 5, 9, 10}:
            return position == 14
        elif attn_0_0_output in {2, 11, 20, 6}:
            return position == 15
        elif attn_0_0_output in {8, 7}:
            return position == 13
        elif attn_0_0_output in {12}:
            return position == 16
        elif attn_0_0_output in {24, 21, 13}:
            return position == 17
        elif attn_0_0_output in {14}:
            return position == 20
        elif attn_0_0_output in {15}:
            return position == 28
        elif attn_0_0_output in {16, 18}:
            return position == 26
        elif attn_0_0_output in {17}:
            return position == 21
        elif attn_0_0_output in {19}:
            return position == 29
        elif attn_0_0_output in {26, 27, 28, 22}:
            return position == 18
        elif attn_0_0_output in {23}:
            return position == 24
        elif attn_0_0_output in {25}:
            return position == 19
        elif attn_0_0_output in {29}:
            return position == 12

    num_attn_1_2_pattern = select(positions, attn_0_0_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_1_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_0_output, position):
        if attn_0_0_output in {0, 1, 6, 7, 8, 9, 27, 29}:
            return position == 16
        elif attn_0_0_output in {2, 20, 5, 23}:
            return position == 15
        elif attn_0_0_output in {3, 4, 28}:
            return position == 14
        elif attn_0_0_output in {10, 11, 12, 14, 21, 24, 25}:
            return position == 19
        elif attn_0_0_output in {13, 15}:
            return position == 26
        elif attn_0_0_output in {16, 17}:
            return position == 29
        elif attn_0_0_output in {18}:
            return position == 27
        elif attn_0_0_output in {26, 19}:
            return position == 20
        elif attn_0_0_output in {22}:
            return position == 22

    num_attn_1_3_pattern = select(positions, attn_0_0_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_2_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_3_output, attn_0_1_output):
        key = (attn_1_3_output, attn_0_1_output)
        return 6

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_3_outputs, attn_0_1_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(num_mlp_0_0_output, num_mlp_0_1_output):
        key = (num_mlp_0_0_output, num_mlp_0_1_output)
        if key in {
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 4),
            (1, 7),
            (1, 9),
            (1, 10),
            (1, 11),
            (1, 13),
            (1, 15),
            (1, 16),
            (1, 17),
            (1, 18),
            (1, 20),
            (1, 21),
            (1, 22),
            (1, 23),
            (1, 24),
            (1, 25),
            (1, 26),
            (1, 27),
            (1, 29),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 4),
            (2, 5),
            (2, 6),
            (2, 7),
            (2, 9),
            (2, 10),
            (2, 11),
            (2, 13),
            (2, 15),
            (2, 16),
            (2, 17),
            (2, 18),
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
            (3, 11),
            (3, 18),
            (3, 22),
            (4, 18),
            (5, 18),
            (6, 18),
            (7, 18),
            (8, 18),
            (10, 18),
            (11, 18),
            (12, 0),
            (12, 1),
            (12, 2),
            (12, 4),
            (12, 6),
            (12, 7),
            (12, 9),
            (12, 10),
            (12, 11),
            (12, 13),
            (12, 15),
            (12, 16),
            (12, 18),
            (12, 20),
            (12, 21),
            (12, 22),
            (12, 23),
            (12, 24),
            (12, 25),
            (12, 27),
            (12, 29),
            (13, 18),
            (14, 18),
            (15, 18),
            (15, 22),
            (16, 18),
            (17, 18),
            (18, 18),
            (19, 18),
            (20, 18),
            (21, 9),
            (21, 11),
            (21, 18),
            (21, 22),
            (22, 18),
            (23, 18),
            (25, 11),
            (25, 18),
            (25, 22),
            (26, 18),
            (27, 18),
            (28, 18),
            (29, 18),
        }:
            return 18
        elif key in {(18, 5), (18, 20), (18, 25)}:
            return 6
        return 22

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(num_mlp_0_0_outputs, num_mlp_0_1_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_2_output, num_attn_0_0_output):
        key = (num_attn_1_2_output, num_attn_0_0_output)
        if key in {
            (22, 1),
            (22, 2),
            (23, 0),
            (23, 1),
            (23, 2),
            (24, 0),
            (24, 1),
            (24, 2),
            (25, 0),
            (25, 1),
            (25, 2),
            (26, 0),
            (26, 1),
            (26, 2),
            (27, 0),
            (27, 1),
            (27, 2),
            (27, 3),
            (28, 0),
            (28, 1),
            (28, 2),
            (28, 3),
            (29, 0),
            (29, 1),
            (29, 2),
            (29, 3),
            (30, 0),
            (30, 1),
            (30, 2),
            (30, 3),
            (31, 0),
            (31, 1),
            (31, 2),
            (31, 3),
            (32, 0),
            (32, 1),
            (32, 2),
            (32, 3),
            (33, 0),
            (33, 1),
            (33, 2),
            (33, 3),
            (34, 0),
            (34, 1),
            (34, 2),
            (34, 3),
            (35, 0),
            (35, 1),
            (35, 2),
            (35, 3),
            (36, 0),
            (36, 1),
            (36, 2),
            (36, 3),
            (37, 0),
            (37, 1),
            (37, 2),
            (37, 3),
            (38, 0),
            (38, 1),
            (38, 2),
            (38, 3),
            (38, 4),
            (39, 0),
            (39, 1),
            (39, 2),
            (39, 3),
            (39, 4),
            (40, 0),
            (40, 1),
            (40, 2),
            (40, 3),
            (40, 4),
            (41, 0),
            (41, 1),
            (41, 2),
            (41, 3),
            (41, 4),
            (42, 0),
            (42, 1),
            (42, 2),
            (42, 3),
            (42, 4),
            (43, 0),
            (43, 1),
            (43, 2),
            (43, 3),
            (43, 4),
            (44, 0),
            (44, 1),
            (44, 2),
            (44, 3),
            (44, 4),
            (45, 0),
            (45, 1),
            (45, 2),
            (45, 3),
            (45, 4),
            (46, 0),
            (46, 1),
            (46, 2),
            (46, 3),
            (46, 4),
            (47, 0),
            (47, 1),
            (47, 2),
            (47, 3),
            (47, 4),
            (48, 0),
            (48, 1),
            (48, 2),
            (48, 3),
            (48, 4),
            (48, 5),
            (49, 0),
            (49, 1),
            (49, 2),
            (49, 3),
            (49, 4),
            (49, 5),
            (50, 0),
            (50, 1),
            (50, 2),
            (50, 3),
            (50, 4),
            (50, 5),
            (51, 0),
            (51, 1),
            (51, 2),
            (51, 3),
            (51, 4),
            (51, 5),
            (52, 0),
            (52, 1),
            (52, 2),
            (52, 3),
            (52, 4),
            (52, 5),
            (53, 0),
            (53, 1),
            (53, 2),
            (53, 3),
            (53, 4),
            (53, 5),
            (54, 0),
            (54, 1),
            (54, 2),
            (54, 3),
            (54, 4),
            (54, 5),
            (55, 0),
            (55, 1),
            (55, 2),
            (55, 3),
            (55, 4),
            (55, 5),
            (56, 0),
            (56, 1),
            (56, 2),
            (56, 3),
            (56, 4),
            (56, 5),
            (57, 0),
            (57, 1),
            (57, 2),
            (57, 3),
            (57, 4),
            (57, 5),
            (58, 0),
            (58, 1),
            (58, 2),
            (58, 3),
            (58, 4),
            (58, 5),
            (59, 0),
            (59, 1),
            (59, 2),
            (59, 3),
            (59, 4),
            (59, 5),
            (59, 6),
        }:
            return 19
        elif key in {
            (0, 2),
            (1, 2),
            (2, 2),
            (3, 2),
            (4, 2),
            (5, 2),
            (6, 2),
            (7, 2),
            (8, 2),
            (9, 2),
            (9, 3),
            (10, 0),
            (10, 1),
            (10, 2),
            (10, 3),
            (11, 0),
            (11, 1),
            (11, 2),
            (11, 3),
            (12, 0),
            (12, 1),
            (12, 2),
            (12, 3),
            (13, 0),
            (13, 1),
            (13, 2),
            (13, 3),
            (14, 0),
            (14, 1),
            (14, 2),
            (14, 3),
            (15, 0),
            (15, 1),
            (15, 2),
            (15, 3),
            (16, 0),
            (16, 1),
            (16, 2),
            (16, 3),
            (17, 0),
            (17, 1),
            (17, 2),
            (17, 3),
            (18, 0),
            (18, 1),
            (18, 2),
            (18, 3),
            (19, 0),
            (19, 1),
            (19, 2),
            (19, 3),
            (20, 0),
            (20, 1),
            (20, 2),
            (20, 3),
            (20, 4),
            (21, 0),
            (21, 1),
            (21, 2),
            (21, 3),
            (21, 4),
            (22, 0),
            (22, 3),
            (22, 4),
            (23, 3),
            (23, 4),
            (24, 3),
            (24, 4),
            (25, 3),
            (25, 4),
            (26, 3),
            (26, 4),
            (27, 4),
            (28, 4),
            (29, 4),
            (30, 4),
            (31, 4),
            (31, 5),
            (32, 4),
            (32, 5),
            (33, 4),
            (33, 5),
            (34, 4),
            (34, 5),
            (35, 4),
            (35, 5),
            (36, 4),
            (36, 5),
            (37, 4),
            (37, 5),
            (38, 5),
            (39, 5),
            (40, 5),
            (41, 5),
            (42, 5),
            (43, 5),
            (43, 6),
            (44, 5),
            (44, 6),
            (45, 5),
            (45, 6),
            (46, 5),
            (46, 6),
            (47, 5),
            (47, 6),
            (48, 6),
            (49, 6),
            (50, 6),
            (51, 6),
            (52, 6),
            (53, 6),
            (54, 6),
            (54, 7),
            (55, 6),
            (55, 7),
            (56, 6),
            (56, 7),
            (57, 6),
            (57, 7),
            (58, 6),
            (58, 7),
            (59, 7),
        }:
            return 29
        elif key in {
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (2, 0),
            (2, 1),
            (3, 0),
            (3, 1),
            (4, 0),
            (4, 1),
            (5, 0),
            (5, 1),
            (6, 0),
            (6, 1),
            (7, 0),
            (7, 1),
            (8, 0),
            (8, 1),
            (9, 0),
            (9, 1),
        }:
            return 23
        return 22

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_0_output, num_attn_1_2_output):
        key = (num_attn_1_0_output, num_attn_1_2_output)
        return 21

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(token, attn_1_1_output):
        if token in {"1", "0"}:
            return attn_1_1_output == 3
        elif token in {"2"}:
            return attn_1_1_output == 29
        elif token in {"3", "4"}:
            return attn_1_1_output == 2
        elif token in {"5"}:
            return attn_1_1_output == 4
        elif token in {"<s>"}:
            return attn_1_1_output == 25

    attn_2_0_pattern = select_closest(attn_1_1_outputs, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_3_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 4
        elif q_position in {1, 2}:
            return k_position == 11
        elif q_position in {25, 3, 20}:
            return k_position == 12
        elif q_position in {11, 4}:
            return k_position == 19
        elif q_position in {12, 5}:
            return k_position == 16
        elif q_position in {6, 10, 13, 14, 15, 19}:
            return k_position == 18
        elif q_position in {7}:
            return k_position == 2
        elif q_position in {8, 9, 16, 18, 28}:
            return k_position == 17
        elif q_position in {17}:
            return k_position == 1
        elif q_position in {21}:
            return k_position == 9
        elif q_position in {26, 29, 22}:
            return k_position == 0
        elif q_position in {23}:
            return k_position == 27
        elif q_position in {24}:
            return k_position == 6
        elif q_position in {27}:
            return k_position == 14

    attn_2_1_pattern = select_closest(positions, positions, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_0_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(attn_0_0_output, attn_1_1_output):
        if attn_0_0_output in {0, 4, 5, 6, 8, 10, 11, 13, 14, 15, 20, 22, 24, 25, 28}:
            return attn_1_1_output == 1
        elif attn_0_0_output in {1}:
            return attn_1_1_output == 21
        elif attn_0_0_output in {2}:
            return attn_1_1_output == 10
        elif attn_0_0_output in {19, 26, 3, 21}:
            return attn_1_1_output == 13
        elif attn_0_0_output in {27, 23, 7}:
            return attn_1_1_output == 0
        elif attn_0_0_output in {16, 9}:
            return attn_1_1_output == 11
        elif attn_0_0_output in {12}:
            return attn_1_1_output == 19
        elif attn_0_0_output in {17}:
            return attn_1_1_output == 9
        elif attn_0_0_output in {18}:
            return attn_1_1_output == 7
        elif attn_0_0_output in {29}:
            return attn_1_1_output == 22

    attn_2_2_pattern = select_closest(attn_1_1_outputs, attn_0_0_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_1_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_3_output, position):
        if attn_0_3_output in {0, 20, 22}:
            return position == 11
        elif attn_0_3_output in {1, 15}:
            return position == 6
        elif attn_0_3_output in {9, 2, 19, 13}:
            return position == 3
        elif attn_0_3_output in {3, 4, 28}:
            return position == 5
        elif attn_0_3_output in {16, 27, 5, 6}:
            return position == 7
        elif attn_0_3_output in {8, 7}:
            return position == 10
        elif attn_0_3_output in {10, 21, 23}:
            return position == 12
        elif attn_0_3_output in {25, 26, 11}:
            return position == 9
        elif attn_0_3_output in {12}:
            return position == 0
        elif attn_0_3_output in {14}:
            return position == 23
        elif attn_0_3_output in {17}:
            return position == 21
        elif attn_0_3_output in {18}:
            return position == 8
        elif attn_0_3_output in {24, 29}:
            return position == 2

    attn_2_3_pattern = select_closest(positions, attn_0_3_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_0_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(num_mlp_1_0_output, position):
        if num_mlp_1_0_output in {0, 1, 8, 17, 21, 25, 26}:
            return position == 15
        elif num_mlp_1_0_output in {24, 2, 23}:
            return position == 13
        elif num_mlp_1_0_output in {3, 12, 13, 14, 15, 27}:
            return position == 16
        elif num_mlp_1_0_output in {4}:
            return position == 18
        elif num_mlp_1_0_output in {5}:
            return position == 27
        elif num_mlp_1_0_output in {6, 7}:
            return position == 23
        elif num_mlp_1_0_output in {9, 18, 28}:
            return position == 12
        elif num_mlp_1_0_output in {10, 19}:
            return position == 28
        elif num_mlp_1_0_output in {16, 11}:
            return position == 14
        elif num_mlp_1_0_output in {20}:
            return position == 29
        elif num_mlp_1_0_output in {22}:
            return position == 20
        elif num_mlp_1_0_output in {29}:
            return position == 17

    num_attn_2_0_pattern = select(positions, num_mlp_1_0_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_0_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(num_mlp_1_0_output, position):
        if num_mlp_1_0_output in {0, 27}:
            return position == 4
        elif num_mlp_1_0_output in {1}:
            return position == 9
        elif num_mlp_1_0_output in {8, 2, 14}:
            return position == 22
        elif num_mlp_1_0_output in {3}:
            return position == 29
        elif num_mlp_1_0_output in {4}:
            return position == 16
        elif num_mlp_1_0_output in {5}:
            return position == 26
        elif num_mlp_1_0_output in {6}:
            return position == 8
        elif num_mlp_1_0_output in {16, 10, 7}:
            return position == 23
        elif num_mlp_1_0_output in {24, 9, 18}:
            return position == 20
        elif num_mlp_1_0_output in {25, 11}:
            return position == 28
        elif num_mlp_1_0_output in {12}:
            return position == 18
        elif num_mlp_1_0_output in {26, 13}:
            return position == 24
        elif num_mlp_1_0_output in {17, 15}:
            return position == 19
        elif num_mlp_1_0_output in {19, 28, 21}:
            return position == 25
        elif num_mlp_1_0_output in {20}:
            return position == 11
        elif num_mlp_1_0_output in {22}:
            return position == 10
        elif num_mlp_1_0_output in {29, 23}:
            return position == 27

    num_attn_2_1_pattern = select(positions, num_mlp_1_0_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_1_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(q_num_mlp_1_0_output, k_num_mlp_1_0_output):
        if q_num_mlp_1_0_output in {0, 2, 6, 11, 21}:
            return k_num_mlp_1_0_output == 7
        elif q_num_mlp_1_0_output in {1}:
            return k_num_mlp_1_0_output == 12
        elif q_num_mlp_1_0_output in {3, 4, 8, 13, 15, 18, 19, 28}:
            return k_num_mlp_1_0_output == 14
        elif q_num_mlp_1_0_output in {5, 7, 10, 12, 14, 29}:
            return k_num_mlp_1_0_output == 29
        elif q_num_mlp_1_0_output in {9}:
            return k_num_mlp_1_0_output == 23
        elif q_num_mlp_1_0_output in {16}:
            return k_num_mlp_1_0_output == 18
        elif q_num_mlp_1_0_output in {17, 27, 25}:
            return k_num_mlp_1_0_output == 13
        elif q_num_mlp_1_0_output in {20}:
            return k_num_mlp_1_0_output == 10
        elif q_num_mlp_1_0_output in {22}:
            return k_num_mlp_1_0_output == 11
        elif q_num_mlp_1_0_output in {24, 23}:
            return k_num_mlp_1_0_output == 9
        elif q_num_mlp_1_0_output in {26}:
            return k_num_mlp_1_0_output == 8

    num_attn_2_2_pattern = select(
        num_mlp_1_0_outputs, num_mlp_1_0_outputs, num_predicate_2_2
    )
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_1_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_0_output, attn_0_0_output):
        if attn_1_0_output in {0, 1, 20, 21, 24, 26, 28, 29}:
            return attn_0_0_output == 0
        elif attn_1_0_output in {25, 2, 10, 12}:
            return attn_0_0_output == 3
        elif attn_1_0_output in {27, 3, 4, 23}:
            return attn_0_0_output == 4
        elif attn_1_0_output in {16, 5}:
            return attn_0_0_output == 5
        elif attn_1_0_output in {6}:
            return attn_0_0_output == 6
        elif attn_1_0_output in {8, 7}:
            return attn_0_0_output == 7
        elif attn_1_0_output in {9}:
            return attn_0_0_output == 8
        elif attn_1_0_output in {11}:
            return attn_0_0_output == 11
        elif attn_1_0_output in {13}:
            return attn_0_0_output == 12
        elif attn_1_0_output in {14, 15}:
            return attn_0_0_output == 13
        elif attn_1_0_output in {17, 19}:
            return attn_0_0_output == 17
        elif attn_1_0_output in {18}:
            return attn_0_0_output == 18
        elif attn_1_0_output in {22}:
            return attn_0_0_output == 2

    num_attn_2_3_pattern = select(attn_0_0_outputs, attn_1_0_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_0_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_0_3_output, mlp_1_0_output):
        key = (attn_0_3_output, mlp_1_0_output)
        return 9

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_0_3_outputs, mlp_1_0_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(num_mlp_0_1_output, attn_2_1_output):
        key = (num_mlp_0_1_output, attn_2_1_output)
        return 23

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(num_mlp_0_1_outputs, attn_2_1_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_2_output):
        key = num_attn_1_2_output
        return 17

    num_mlp_2_0_outputs = [num_mlp_2_0(k0) for k0 in num_attn_1_2_outputs]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_0_output, num_attn_2_2_output):
        key = (num_attn_2_0_output, num_attn_2_2_output)
        return 18

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_0_outputs, num_attn_2_2_outputs)
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


print(
    run(
        [
            "<s>",
            "5",
            "1",
            "0",
            "0",
            "2",
            "1",
            "2",
            "4",
            "5",
            "1",
            "0",
            "4",
            "2",
            "4",
            "2",
        ]
    )
)
