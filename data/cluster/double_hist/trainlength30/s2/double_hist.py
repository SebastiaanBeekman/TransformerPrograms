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
        "output/length/rasp/double_hist/trainlength30/s2/double_hist_weights.csv",
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
            return k_token == "5"
        elif q_token in {"1", "3"}:
            return k_token == "2"
        elif q_token in {"2"}:
            return k_token == "0"
        elif q_token in {"4", "5"}:
            return k_token == "3"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_0_0_pattern = select_closest(tokens, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, positions)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_token, k_token):
        if q_token in {"1", "0"}:
            return k_token == "2"
        elif q_token in {"4", "3", "2"}:
            return k_token == "1"
        elif q_token in {"5"}:
            return k_token == "3"
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
            return k_token == "3"
        elif q_token in {"1"}:
            return k_token == "0"
        elif q_token in {"5", "2"}:
            return k_token == "4"
        elif q_token in {"3"}:
            return k_token == "5"
        elif q_token in {"4"}:
            return k_token == "1"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_0_2_pattern = select_closest(tokens, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, positions)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "3"
        elif q_token in {"1"}:
            return k_token == "0"
        elif q_token in {"4", "3", "2"}:
            return k_token == "5"
        elif q_token in {"5"}:
            return k_token == "4"
        elif q_token in {"<s>"}:
            return k_token == "2"

    attn_0_3_pattern = select_closest(tokens, tokens, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, positions)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 2
        elif q_position in {32, 1, 36, 7, 31}:
            return k_position == 27
        elif q_position in {2, 11, 12}:
            return k_position == 28
        elif q_position in {8, 3, 6, 15}:
            return k_position == 25
        elif q_position in {4, 30}:
            return k_position == 13
        elif q_position in {18, 5}:
            return k_position == 39
        elif q_position in {9}:
            return k_position == 33
        elif q_position in {10, 13, 38}:
            return k_position == 29
        elif q_position in {14}:
            return k_position == 20
        elif q_position in {16, 24, 20}:
            return k_position == 31
        elif q_position in {17, 37, 39}:
            return k_position == 34
        elif q_position in {34, 19}:
            return k_position == 38
        elif q_position in {21}:
            return k_position == 37
        elif q_position in {22, 23}:
            return k_position == 35
        elif q_position in {25}:
            return k_position == 23
        elif q_position in {26, 29}:
            return k_position == 32
        elif q_position in {27}:
            return k_position == 36
        elif q_position in {28}:
            return k_position == 7
        elif q_position in {33}:
            return k_position == 30
        elif q_position in {35}:
            return k_position == 15

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
            return k_token == "<pad>"

    num_attn_0_1_pattern = select(tokens, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_token, k_token):
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

    num_attn_0_2_pattern = select(tokens, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {0, 22}:
            return k_position == 27
        elif q_position in {1}:
            return k_position == 19
        elif q_position in {2, 3, 37, 30}:
            return k_position == 20
        elif q_position in {4, 5, 6, 12, 13}:
            return k_position == 22
        elif q_position in {9, 10, 17, 7}:
            return k_position == 21
        elif q_position in {8, 11}:
            return k_position == 18
        elif q_position in {16, 18, 14}:
            return k_position == 25
        elif q_position in {15}:
            return k_position == 30
        elif q_position in {35, 19}:
            return k_position == 26
        elif q_position in {27, 20}:
            return k_position == 28
        elif q_position in {21}:
            return k_position == 14
        elif q_position in {33, 31, 23}:
            return k_position == 29
        elif q_position in {24, 25}:
            return k_position == 36
        elif q_position in {26}:
            return k_position == 34
        elif q_position in {28, 38}:
            return k_position == 37
        elif q_position in {29}:
            return k_position == 16
        elif q_position in {32}:
            return k_position == 35
        elif q_position in {34}:
            return k_position == 33
        elif q_position in {36}:
            return k_position == 31
        elif q_position in {39}:
            return k_position == 38

    num_attn_0_3_pattern = select(positions, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_0_output):
        key = attn_0_0_output
        return 7

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in attn_0_0_outputs]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(token, attn_0_3_output):
        key = (token, attn_0_3_output)
        return 35

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(tokens, attn_0_3_outputs)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output):
        key = num_attn_0_0_output
        return 31

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_0_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_0_output, num_attn_0_3_output):
        key = (num_attn_0_0_output, num_attn_0_3_output)
        return 3

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_token, k_token):
        if q_token in {"0", "3"}:
            return k_token == "1"
        elif q_token in {"1"}:
            return k_token == "5"
        elif q_token in {"2"}:
            return k_token == "4"
        elif q_token in {"4"}:
            return k_token == "0"
        elif q_token in {"<s>", "5"}:
            return k_token == "2"

    attn_1_0_pattern = select_closest(tokens, tokens, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, positions)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(attn_0_3_output, token):
        if attn_0_3_output in {0, 32, 2, 33, 35, 6, 38, 8, 19, 22, 24}:
            return token == ""
        elif attn_0_3_output in {1, 7, 15, 16, 18, 21, 30}:
            return token == "2"
        elif attn_0_3_output in {3, 36, 39, 12, 14, 17, 31}:
            return token == "3"
        elif attn_0_3_output in {4, 20}:
            return token == "4"
        elif attn_0_3_output in {10, 29, 34, 5}:
            return token == "1"
        elif attn_0_3_output in {37, 9, 23, 25, 26, 28}:
            return token == "5"
        elif attn_0_3_output in {27, 11, 13}:
            return token == "0"

    attn_1_1_pattern = select_closest(tokens, attn_0_3_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_3_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_token, k_token):
        if q_token in {"1", "0", "3"}:
            return k_token == "4"
        elif q_token in {"5", "2"}:
            return k_token == "1"
        elif q_token in {"4"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == "0"

    attn_1_2_pattern = select_closest(tokens, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, positions)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "4"
        elif q_token in {"1"}:
            return k_token == "5"
        elif q_token in {"2"}:
            return k_token == "3"
        elif q_token in {"3", "5"}:
            return k_token == "0"
        elif q_token in {"4"}:
            return k_token == "2"
        elif q_token in {"<s>"}:
            return k_token == "1"

    attn_1_3_pattern = select_closest(tokens, tokens, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, positions)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_2_output, position):
        if attn_0_2_output in {0}:
            return position == 3
        elif attn_0_2_output in {1}:
            return position == 2
        elif attn_0_2_output in {33, 2}:
            return position == 1
        elif attn_0_2_output in {3, 4, 5}:
            return position == 4
        elif attn_0_2_output in {6}:
            return position == 5
        elif attn_0_2_output in {9, 7}:
            return position == 6
        elif attn_0_2_output in {8, 10, 11, 12, 14, 19}:
            return position == 7
        elif attn_0_2_output in {28, 13}:
            return position == 11
        elif attn_0_2_output in {15}:
            return position == 9
        elif attn_0_2_output in {16, 17, 22}:
            return position == 12
        elif attn_0_2_output in {18, 23}:
            return position == 13
        elif attn_0_2_output in {20}:
            return position == 19
        elif attn_0_2_output in {24, 21}:
            return position == 17
        elif attn_0_2_output in {25}:
            return position == 24
        elif attn_0_2_output in {26}:
            return position == 18
        elif attn_0_2_output in {27}:
            return position == 21
        elif attn_0_2_output in {29}:
            return position == 26
        elif attn_0_2_output in {32, 34, 35, 36, 37, 38, 39, 30, 31}:
            return position == 0

    num_attn_1_0_pattern = select(positions, attn_0_2_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_1_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_3_output, attn_0_1_output):
        if attn_0_3_output in {0, 33, 32, 34, 39, 30}:
            return attn_0_1_output == 0
        elif attn_0_3_output in {1}:
            return attn_0_1_output == 1
        elif attn_0_3_output in {2}:
            return attn_0_1_output == 2
        elif attn_0_3_output in {35, 3}:
            return attn_0_1_output == 3
        elif attn_0_3_output in {4}:
            return attn_0_1_output == 4
        elif attn_0_3_output in {5}:
            return attn_0_1_output == 5
        elif attn_0_3_output in {6}:
            return attn_0_1_output == 6
        elif attn_0_3_output in {10, 27, 7}:
            return attn_0_1_output == 37
        elif attn_0_3_output in {8, 18, 15}:
            return attn_0_1_output == 31
        elif attn_0_3_output in {9, 37}:
            return attn_0_1_output == 34
        elif attn_0_3_output in {38, 17, 11, 22}:
            return attn_0_1_output == 32
        elif attn_0_3_output in {20, 12, 13, 23}:
            return attn_0_1_output == 35
        elif attn_0_3_output in {19, 36, 14}:
            return attn_0_1_output == 36
        elif attn_0_3_output in {16, 31}:
            return attn_0_1_output == 39
        elif attn_0_3_output in {28, 21}:
            return attn_0_1_output == 38
        elif attn_0_3_output in {24, 25, 26}:
            return attn_0_1_output == 33
        elif attn_0_3_output in {29}:
            return attn_0_1_output == 27

    num_attn_1_1_pattern = select(attn_0_1_outputs, attn_0_3_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_2_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_3_output, token):
        if attn_0_3_output in {
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
        }:
            return token == ""

    num_attn_1_2_pattern = select(tokens, attn_0_3_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_0_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_2_output, position):
        if attn_0_2_output in {0}:
            return position == 11
        elif attn_0_2_output in {32, 1}:
            return position == 31
        elif attn_0_2_output in {8, 2}:
            return position == 19
        elif attn_0_2_output in {3}:
            return position == 22
        elif attn_0_2_output in {4}:
            return position == 9
        elif attn_0_2_output in {35, 36, 5}:
            return position == 28
        elif attn_0_2_output in {6}:
            return position == 23
        elif attn_0_2_output in {16, 7}:
            return position == 36
        elif attn_0_2_output in {38, 9, 10, 13, 14}:
            return position == 29
        elif attn_0_2_output in {26, 11, 30}:
            return position == 37
        elif attn_0_2_output in {12, 15}:
            return position == 38
        elif attn_0_2_output in {34, 37, 17, 24, 27}:
            return position == 39
        elif attn_0_2_output in {18, 20, 21}:
            return position == 30
        elif attn_0_2_output in {19, 29, 22, 23}:
            return position == 32
        elif attn_0_2_output in {25, 33}:
            return position == 35
        elif attn_0_2_output in {28, 39}:
            return position == 34
        elif attn_0_2_output in {31}:
            return position == 25

    num_attn_1_3_pattern = select(positions, attn_0_2_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_3_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_0_output):
        key = attn_1_0_output
        return 36

    mlp_1_0_outputs = [mlp_1_0(k0) for k0 in attn_1_0_outputs]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_0_output, attn_1_2_output):
        key = (attn_1_0_output, attn_1_2_output)
        return 10

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_0_outputs, attn_1_2_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_2_output, num_attn_0_0_output):
        key = (num_attn_1_2_output, num_attn_0_0_output)
        if key in {
            (16, 0),
            (17, 0),
            (18, 0),
            (19, 0),
            (20, 0),
            (21, 0),
            (22, 0),
            (23, 0),
            (24, 0),
            (25, 0),
            (26, 0),
            (27, 0),
            (28, 0),
            (29, 0),
            (30, 0),
            (30, 1),
            (31, 0),
            (31, 1),
            (32, 1),
            (33, 1),
            (34, 1),
            (35, 1),
            (36, 1),
            (37, 1),
            (38, 1),
            (39, 1),
            (40, 1),
            (41, 1),
            (42, 1),
            (43, 1),
            (43, 2),
            (44, 1),
            (44, 2),
            (45, 2),
            (46, 2),
            (47, 2),
            (48, 2),
            (49, 2),
            (50, 2),
            (51, 2),
            (52, 2),
            (53, 2),
            (54, 2),
            (55, 2),
            (56, 2),
            (57, 2),
            (57, 3),
            (58, 2),
            (58, 3),
            (59, 3),
            (60, 3),
            (61, 3),
            (62, 3),
            (63, 3),
            (64, 3),
            (65, 3),
            (66, 3),
            (67, 3),
            (68, 3),
            (69, 3),
            (70, 3),
            (70, 4),
            (71, 3),
            (71, 4),
            (72, 3),
            (72, 4),
            (73, 4),
            (74, 4),
            (75, 4),
            (76, 4),
            (77, 4),
            (78, 4),
            (79, 4),
        }:
            return 16
        elif key in {
            (32, 0),
            (33, 0),
            (34, 0),
            (35, 0),
            (36, 0),
            (37, 0),
            (38, 0),
            (39, 0),
            (40, 0),
            (41, 0),
            (42, 0),
            (43, 0),
            (44, 0),
            (45, 0),
            (45, 1),
            (46, 0),
            (46, 1),
            (47, 0),
            (47, 1),
            (48, 0),
            (48, 1),
            (49, 0),
            (49, 1),
            (50, 0),
            (50, 1),
            (51, 0),
            (51, 1),
            (52, 0),
            (52, 1),
            (53, 1),
            (54, 1),
            (55, 1),
            (56, 1),
            (57, 1),
            (58, 1),
            (59, 1),
            (59, 2),
            (60, 1),
            (60, 2),
            (61, 1),
            (61, 2),
            (62, 1),
            (62, 2),
            (63, 1),
            (63, 2),
            (64, 2),
            (65, 2),
            (66, 2),
            (67, 2),
            (68, 2),
            (69, 2),
            (70, 2),
            (71, 2),
            (72, 2),
            (73, 2),
            (73, 3),
            (74, 2),
            (74, 3),
            (75, 3),
            (76, 3),
            (77, 3),
            (78, 3),
            (79, 3),
        }:
            return 39
        elif key in {
            (53, 0),
            (54, 0),
            (55, 0),
            (56, 0),
            (57, 0),
            (58, 0),
            (59, 0),
            (60, 0),
            (61, 0),
            (62, 0),
            (63, 0),
            (64, 0),
            (64, 1),
            (65, 0),
            (65, 1),
            (66, 0),
            (66, 1),
            (67, 0),
            (67, 1),
            (68, 0),
            (68, 1),
            (69, 0),
            (69, 1),
            (70, 0),
            (70, 1),
            (71, 0),
            (71, 1),
            (72, 0),
            (72, 1),
            (73, 0),
            (73, 1),
            (74, 0),
            (74, 1),
            (75, 0),
            (75, 1),
            (75, 2),
            (76, 0),
            (76, 1),
            (76, 2),
            (77, 0),
            (77, 1),
            (77, 2),
            (78, 0),
            (78, 1),
            (78, 2),
            (79, 0),
            (79, 1),
            (79, 2),
        }:
            return 37
        return 2

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_1_output, num_attn_1_2_output):
        key = (num_attn_1_1_output, num_attn_1_2_output)
        return 2

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "4"
        elif q_token in {"1"}:
            return k_token == "2"
        elif q_token in {"2"}:
            return k_token == ""
        elif q_token in {"3"}:
            return k_token == "0"
        elif q_token in {"4", "5"}:
            return k_token == "3"
        elif q_token in {"<s>"}:
            return k_token == "5"

    attn_2_0_pattern = select_closest(tokens, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_3_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_0_1_output, position):
        if attn_0_1_output in {0, 2, 19, 14}:
            return position == 7
        elif attn_0_1_output in {27, 1, 3, 5}:
            return position == 6
        elif attn_0_1_output in {4}:
            return position == 5
        elif attn_0_1_output in {25, 6}:
            return position == 11
        elif attn_0_1_output in {26, 7}:
            return position == 13
        elif attn_0_1_output in {8, 9}:
            return position == 1
        elif attn_0_1_output in {10, 11}:
            return position == 2
        elif attn_0_1_output in {12, 13, 15}:
            return position == 9
        elif attn_0_1_output in {16}:
            return position == 3
        elif attn_0_1_output in {32, 17, 34, 31}:
            return position == 14
        elif attn_0_1_output in {18, 22}:
            return position == 10
        elif attn_0_1_output in {33, 35, 20, 30}:
            return position == 17
        elif attn_0_1_output in {37, 21}:
            return position == 19
        elif attn_0_1_output in {38, 23}:
            return position == 15
        elif attn_0_1_output in {24, 36, 39}:
            return position == 12
        elif attn_0_1_output in {28}:
            return position == 24
        elif attn_0_1_output in {29}:
            return position == 16

    attn_2_1_pattern = select_closest(positions, attn_0_1_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, positions)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_position, k_position):
        if q_position in {0}:
            return k_position == 3
        elif q_position in {1, 36, 23}:
            return k_position == 24
        elif q_position in {2}:
            return k_position == 26
        elif q_position in {8, 24, 3, 27}:
            return k_position == 29
        elif q_position in {4, 15}:
            return k_position == 0
        elif q_position in {9, 28, 5, 17}:
            return k_position == 12
        elif q_position in {6}:
            return k_position == 20
        elif q_position in {7}:
            return k_position == 23
        elif q_position in {10}:
            return k_position == 13
        elif q_position in {11}:
            return k_position == 22
        elif q_position in {12}:
            return k_position == 7
        elif q_position in {13}:
            return k_position == 2
        elif q_position in {14}:
            return k_position == 10
        elif q_position in {16, 33}:
            return k_position == 18
        elif q_position in {18}:
            return k_position == 8
        elif q_position in {19, 39}:
            return k_position == 9
        elif q_position in {20}:
            return k_position == 21
        elif q_position in {26, 34, 21}:
            return k_position == 1
        elif q_position in {22}:
            return k_position == 27
        elif q_position in {32, 25, 29}:
            return k_position == 17
        elif q_position in {35, 37, 30}:
            return k_position == 4
        elif q_position in {31}:
            return k_position == 15
        elif q_position in {38}:
            return k_position == 36

    attn_2_2_pattern = select_closest(positions, positions, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_2_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_token, k_token):
        if q_token in {"1", "0"}:
            return k_token == "3"
        elif q_token in {"4", "2"}:
            return k_token == "0"
        elif q_token in {"3"}:
            return k_token == "4"
        elif q_token in {"5"}:
            return k_token == "1"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_3_pattern = select_closest(tokens, tokens, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_0_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_2_output, token):
        if attn_1_2_output in {
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            17,
            18,
            20,
            21,
            22,
            23,
            26,
            27,
            29,
            30,
            32,
            33,
            35,
            38,
            39,
        }:
            return token == ""
        elif attn_1_2_output in {34, 36, 37, 8, 16, 19, 24, 25, 28, 31}:
            return token == "<pad>"

    num_attn_2_0_pattern = select(tokens, attn_1_2_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_3_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_0_3_output, position):
        if attn_0_3_output in {0, 1}:
            return position == 14
        elif attn_0_3_output in {2}:
            return position == 13
        elif attn_0_3_output in {3, 6}:
            return position == 21
        elif attn_0_3_output in {37, 4, 5, 39}:
            return position == 22
        elif attn_0_3_output in {20, 7}:
            return position == 30
        elif attn_0_3_output in {8, 26, 14}:
            return position == 29
        elif attn_0_3_output in {9}:
            return position == 27
        elif attn_0_3_output in {10, 19, 15}:
            return position == 28
        elif attn_0_3_output in {11, 12, 13}:
            return position == 25
        elif attn_0_3_output in {16, 25, 22}:
            return position == 34
        elif attn_0_3_output in {17}:
            return position == 38
        elif attn_0_3_output in {18, 27}:
            return position == 36
        elif attn_0_3_output in {35, 21, 30}:
            return position == 37
        elif attn_0_3_output in {28, 29, 23}:
            return position == 35
        elif attn_0_3_output in {24, 38}:
            return position == 31
        elif attn_0_3_output in {31}:
            return position == 33
        elif attn_0_3_output in {32, 36}:
            return position == 26
        elif attn_0_3_output in {33, 34}:
            return position == 32

    num_attn_2_1_pattern = select(positions, attn_0_3_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_3_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_0_output, attn_1_3_output):
        if attn_1_0_output in {0, 2, 3, 4, 34, 35, 38}:
            return attn_1_3_output == 0
        elif attn_1_0_output in {1, 10}:
            return attn_1_3_output == 39
        elif attn_1_0_output in {26, 27, 5}:
            return attn_1_3_output == 32
        elif attn_1_0_output in {25, 21, 6}:
            return attn_1_3_output == 34
        elif attn_1_0_output in {7, 18, 19, 22, 30}:
            return attn_1_3_output == 33
        elif attn_1_0_output in {32, 33, 8, 9, 14}:
            return attn_1_3_output == 37
        elif attn_1_0_output in {11, 13, 23}:
            return attn_1_3_output == 30
        elif attn_1_0_output in {16, 37, 12, 28}:
            return attn_1_3_output == 36
        elif attn_1_0_output in {17, 36, 29, 15}:
            return attn_1_3_output == 35
        elif attn_1_0_output in {24, 20, 39, 31}:
            return attn_1_3_output == 31

    num_attn_2_2_pattern = select(attn_1_3_outputs, attn_1_0_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_1_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(mlp_1_0_output, attn_0_0_output):
        if mlp_1_0_output in {0, 34, 11, 13, 23, 26}:
            return attn_0_0_output == 38
        elif mlp_1_0_output in {1, 4, 6, 18, 20}:
            return attn_0_0_output == 31
        elif mlp_1_0_output in {24, 2, 37, 29}:
            return attn_0_0_output == 37
        elif mlp_1_0_output in {3, 39, 31}:
            return attn_0_0_output == 39
        elif mlp_1_0_output in {17, 5, 7}:
            return attn_0_0_output == 34
        elif mlp_1_0_output in {33, 38, 8, 15, 16, 27, 30}:
            return attn_0_0_output == 36
        elif mlp_1_0_output in {32, 9, 28, 25}:
            return attn_0_0_output == 35
        elif mlp_1_0_output in {35, 10, 19, 36}:
            return attn_0_0_output == 32
        elif mlp_1_0_output in {12, 21, 22}:
            return attn_0_0_output == 30
        elif mlp_1_0_output in {14}:
            return attn_0_0_output == 33

    num_attn_2_3_pattern = select(attn_0_0_outputs, mlp_1_0_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_1_2_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(mlp_1_0_output, attn_2_2_output):
        key = (mlp_1_0_output, attn_2_2_output)
        if key in {
            (3, 3),
            (3, 4),
            (3, 7),
            (3, 8),
            (3, 15),
            (3, 16),
            (3, 18),
            (3, 19),
            (3, 21),
            (3, 23),
            (3, 31),
            (3, 35),
            (3, 39),
            (4, 15),
            (4, 19),
            (4, 21),
            (5, 15),
            (5, 16),
            (5, 19),
            (5, 21),
            (6, 15),
            (6, 21),
            (7, 15),
            (7, 16),
            (7, 19),
            (7, 21),
            (8, 3),
            (8, 4),
            (8, 7),
            (8, 8),
            (8, 15),
            (8, 16),
            (8, 18),
            (8, 19),
            (8, 21),
            (8, 23),
            (8, 31),
            (8, 35),
            (8, 39),
            (9, 15),
            (9, 16),
            (9, 19),
            (9, 21),
            (10, 15),
            (10, 16),
            (10, 19),
            (10, 21),
            (11, 15),
            (11, 21),
            (12, 15),
            (12, 16),
            (12, 19),
            (12, 21),
            (13, 3),
            (13, 4),
            (13, 7),
            (13, 15),
            (13, 16),
            (13, 18),
            (13, 19),
            (13, 21),
            (13, 23),
            (14, 15),
            (14, 16),
            (14, 19),
            (14, 21),
            (18, 3),
            (18, 4),
            (18, 15),
            (18, 16),
            (18, 18),
            (18, 19),
            (18, 21),
            (18, 23),
            (19, 3),
            (19, 4),
            (19, 7),
            (19, 15),
            (19, 16),
            (19, 18),
            (19, 19),
            (19, 21),
            (19, 23),
            (21, 3),
            (21, 4),
            (21, 15),
            (21, 16),
            (21, 18),
            (21, 19),
            (21, 21),
            (21, 23),
            (22, 3),
            (22, 4),
            (22, 7),
            (22, 8),
            (22, 15),
            (22, 16),
            (22, 18),
            (22, 19),
            (22, 21),
            (22, 22),
            (22, 23),
            (22, 31),
            (22, 32),
            (22, 33),
            (22, 35),
            (22, 36),
            (22, 37),
            (22, 38),
            (22, 39),
            (23, 15),
            (23, 19),
            (23, 21),
            (24, 15),
            (24, 16),
            (24, 19),
            (24, 21),
            (27, 15),
            (27, 16),
            (27, 19),
            (27, 21),
            (28, 15),
            (28, 16),
            (28, 19),
            (28, 21),
            (31, 15),
            (31, 16),
            (31, 19),
            (31, 21),
            (32, 15),
            (32, 16),
            (32, 19),
            (32, 21),
            (33, 3),
            (33, 4),
            (33, 7),
            (33, 15),
            (33, 16),
            (33, 18),
            (33, 19),
            (33, 21),
            (33, 23),
            (33, 39),
            (34, 15),
            (34, 21),
            (35, 15),
            (35, 16),
            (35, 19),
            (35, 21),
            (36, 15),
            (36, 16),
            (36, 19),
            (36, 21),
            (37, 15),
            (37, 19),
            (37, 21),
            (38, 3),
            (38, 4),
            (38, 7),
            (38, 8),
            (38, 15),
            (38, 16),
            (38, 18),
            (38, 19),
            (38, 21),
            (38, 23),
            (38, 31),
            (38, 35),
            (38, 39),
            (39, 15),
            (39, 16),
            (39, 19),
            (39, 21),
        }:
            return 19
        return 23

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(mlp_1_0_outputs, attn_2_2_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(mlp_1_0_output, num_mlp_0_1_output):
        key = (mlp_1_0_output, num_mlp_0_1_output)
        return 25

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(mlp_1_0_outputs, num_mlp_0_1_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_0_output, num_attn_2_0_output):
        key = (num_attn_1_0_output, num_attn_2_0_output)
        return 37

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_2_0_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_2_output, num_attn_1_3_output):
        key = (num_attn_1_2_output, num_attn_1_3_output)
        return 20

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_1_3_outputs)
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


print(run(["<s>", "5", "0", "3", "2", "3", "0", "2", "1", "3"]))
