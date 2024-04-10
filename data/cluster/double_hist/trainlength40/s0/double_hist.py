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
        "output/length/rasp/double_hist/trainlength40/s0/double_hist_weights.csv",
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
        if q_token in {"3", "0"}:
            return k_token == "4"
        elif q_token in {"4", "1", "<s>"}:
            return k_token == "2"
        elif q_token in {"2"}:
            return k_token == "3"
        elif q_token in {"5"}:
            return k_token == "0"

    attn_0_0_pattern = select_closest(tokens, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, positions)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 3
        elif q_position in {1, 2, 3, 5}:
            return k_position == 6
        elif q_position in {4}:
            return k_position == 5
        elif q_position in {19, 6, 7}:
            return k_position == 7
        elif q_position in {8, 9, 10, 12, 13, 14}:
            return k_position == 8
        elif q_position in {11, 30, 15}:
            return k_position == 11
        elif q_position in {40, 16, 17, 18, 48, 25}:
            return k_position == 9
        elif q_position in {20}:
            return k_position == 13
        elif q_position in {34, 21, 22, 27, 28, 29}:
            return k_position == 14
        elif q_position in {24, 23}:
            return k_position == 15
        elif q_position in {26}:
            return k_position == 10
        elif q_position in {33, 31}:
            return k_position == 16
        elif q_position in {32}:
            return k_position == 19
        elif q_position in {35}:
            return k_position == 17
        elif q_position in {36, 38}:
            return k_position == 23
        elif q_position in {37}:
            return k_position == 20
        elif q_position in {45, 39}:
            return k_position == 18
        elif q_position in {41}:
            return k_position == 26
        elif q_position in {42, 44, 46, 47, 49}:
            return k_position == 4
        elif q_position in {43}:
            return k_position == 12

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, positions)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_token, k_token):
        if q_token in {"2", "3", "0"}:
            return k_token == "1"
        elif q_token in {"1"}:
            return k_token == "4"
        elif q_token in {"4", "<s>"}:
            return k_token == "5"
        elif q_token in {"5"}:
            return k_token == "3"

    attn_0_2_pattern = select_closest(tokens, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, positions)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_token, k_token):
        if q_token in {"4", "2", "5", "0"}:
            return k_token == "3"
        elif q_token in {"1", "3"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == "1"

    attn_0_3_pattern = select_closest(tokens, tokens, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_token, k_token):
        if q_token in {"<s>", "4", "2", "5", "1", "3", "0"}:
            return k_token == ""

    num_attn_0_0_pattern = select(tokens, tokens, num_predicate_0_0)
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
    def num_predicate_0_2(position, token):
        if position in {
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
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
        }:
            return token == ""

    num_attn_0_2_pattern = select(tokens, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 13
        elif q_position in {1, 2, 3, 4}:
            return k_position == 24
        elif q_position in {9, 5, 7}:
            return k_position == 30
        elif q_position in {16, 10, 28, 6}:
            return k_position == 36
        elif q_position in {8}:
            return k_position == 26
        elif q_position in {26, 11, 13}:
            return k_position == 32
        elif q_position in {34, 12, 20}:
            return k_position == 37
        elif q_position in {14}:
            return k_position == 38
        elif q_position in {36, 38, 42, 15, 23}:
            return k_position == 45
        elif q_position in {17}:
            return k_position == 49
        elif q_position in {24, 33, 18}:
            return k_position == 46
        elif q_position in {48, 19}:
            return k_position == 34
        elif q_position in {32, 45, 37, 21}:
            return k_position == 41
        elif q_position in {22}:
            return k_position == 47
        elif q_position in {25, 30, 49}:
            return k_position == 43
        elif q_position in {27}:
            return k_position == 48
        elif q_position in {41, 29, 31}:
            return k_position == 40
        elif q_position in {35}:
            return k_position == 44
        elif q_position in {39}:
            return k_position == 29
        elif q_position in {40}:
            return k_position == 0
        elif q_position in {43}:
            return k_position == 33
        elif q_position in {44, 46}:
            return k_position == 8
        elif q_position in {47}:
            return k_position == 35

    num_attn_0_3_pattern = select(positions, positions, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_3_output):
        key = attn_0_3_output
        if key in {"", "<pad>"}:
            return 34
        return 16

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in attn_0_3_outputs]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_1_output, token):
        key = (attn_0_1_output, token)
        return 9

    mlp_0_1_outputs = [mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_1_outputs, tokens)]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_2_output, num_attn_0_1_output):
        key = (num_attn_0_2_output, num_attn_0_1_output)
        return 1

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_2_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(one, num_attn_0_3_output):
        key = (one, num_attn_0_3_output)
        return 3

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1) for k0, k1 in zip(ones, num_attn_0_3_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_token, k_token):
        if q_token in {"2", "0"}:
            return k_token == "3"
        elif q_token in {"1"}:
            return k_token == "5"
        elif q_token in {"3", "5"}:
            return k_token == "1"
        elif q_token in {"4"}:
            return k_token == "2"
        elif q_token in {"<s>"}:
            return k_token == "0"

    attn_1_0_pattern = select_closest(tokens, tokens, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, positions)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_token, k_token):
        if q_token in {"2", "0"}:
            return k_token == "4"
        elif q_token in {"1"}:
            return k_token == "5"
        elif q_token in {"3"}:
            return k_token == "0"
        elif q_token in {"4"}:
            return k_token == "2"
        elif q_token in {"5"}:
            return k_token == "3"
        elif q_token in {"<s>"}:
            return k_token == "1"

    attn_1_1_pattern = select_closest(tokens, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_0_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_token, k_token):
        if q_token in {"<s>", "4", "2", "1", "3", "0"}:
            return k_token == ""
        elif q_token in {"5"}:
            return k_token == "2"

    attn_1_2_pattern = select_closest(tokens, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_2_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_1_output, position):
        if attn_0_1_output in {0, 3}:
            return position == 4
        elif attn_0_1_output in {40, 1}:
            return position == 34
        elif attn_0_1_output in {2}:
            return position == 48
        elif attn_0_1_output in {4}:
            return position == 5
        elif attn_0_1_output in {44, 5, 6}:
            return position == 6
        elif attn_0_1_output in {8, 7}:
            return position == 8
        elif attn_0_1_output in {9, 10, 47}:
            return position == 10
        elif attn_0_1_output in {11}:
            return position == 11
        elif attn_0_1_output in {12}:
            return position == 12
        elif attn_0_1_output in {41, 13}:
            return position == 13
        elif attn_0_1_output in {24, 14}:
            return position == 14
        elif attn_0_1_output in {48, 15}:
            return position == 7
        elif attn_0_1_output in {16, 20}:
            return position == 25
        elif attn_0_1_output in {17, 18}:
            return position == 16
        elif attn_0_1_output in {34, 43, 49, 19, 28, 31}:
            return position == 31
        elif attn_0_1_output in {37, 21}:
            return position == 22
        elif attn_0_1_output in {42, 36, 22, 23}:
            return position == 24
        elif attn_0_1_output in {25, 45}:
            return position == 18
        elif attn_0_1_output in {26}:
            return position == 29
        elif attn_0_1_output in {27}:
            return position == 41
        elif attn_0_1_output in {29}:
            return position == 23
        elif attn_0_1_output in {30}:
            return position == 37
        elif attn_0_1_output in {32, 35}:
            return position == 30
        elif attn_0_1_output in {33}:
            return position == 21
        elif attn_0_1_output in {38}:
            return position == 32
        elif attn_0_1_output in {39}:
            return position == 38
        elif attn_0_1_output in {46}:
            return position == 17

    attn_1_3_pattern = select_closest(positions, attn_0_1_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_1_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_3_output, token):
        if attn_0_3_output in {"0"}:
            return token == "0"
        elif attn_0_3_output in {"1"}:
            return token == "1"
        elif attn_0_3_output in {"2"}:
            return token == "2"
        elif attn_0_3_output in {"3", "<s>", "5"}:
            return token == ""
        elif attn_0_3_output in {"4"}:
            return token == "4"

    num_attn_1_0_pattern = select(tokens, attn_0_3_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, ones)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_3_output, token):
        if attn_0_3_output in {"<s>", "4", "2", "5", "1", "3", "0"}:
            return token == ""

    num_attn_1_1_pattern = select(tokens, attn_0_3_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_2_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(mlp_0_1_output, token):
        if mlp_0_1_output in {
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
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
        }:
            return token == ""

    num_attn_1_2_pattern = select(tokens, mlp_0_1_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, ones)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(num_mlp_0_1_output, token):
        if num_mlp_0_1_output in {
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
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
        }:
            return token == ""

    num_attn_1_3_pattern = select(tokens, num_mlp_0_1_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_0_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_0_output, attn_1_2_output):
        key = (attn_0_0_output, attn_1_2_output)
        return 2

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_1_2_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_0_3_output, token):
        key = (attn_0_3_output, token)
        return 16

    mlp_1_1_outputs = [mlp_1_1(k0, k1) for k0, k1 in zip(attn_0_3_outputs, tokens)]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_0_1_output, num_attn_0_3_output):
        key = (num_attn_0_1_output, num_attn_0_3_output)
        return 9

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_1_output, num_attn_0_1_output):
        key = (num_attn_1_1_output, num_attn_0_1_output)
        return 30

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_token, k_token):
        if q_token in {"2", "1", "4", "0"}:
            return k_token == "3"
        elif q_token in {"3", "<s>"}:
            return k_token == "4"
        elif q_token in {"5"}:
            return k_token == "0"

    attn_2_0_pattern = select_closest(tokens, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_0_0_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_token, k_token):
        if q_token in {"4", "2", "3", "0"}:
            return k_token == "1"
        elif q_token in {"1"}:
            return k_token == "2"
        elif q_token in {"5"}:
            return k_token == "4"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_1_pattern = select_closest(tokens, tokens, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_3_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_token, k_token):
        if q_token in {"4", "2", "3", "0"}:
            return k_token == "5"
        elif q_token in {"1"}:
            return k_token == "2"
        elif q_token in {"5"}:
            return k_token == "<s>"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_2_pattern = select_closest(tokens, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_0_0_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_token, k_token):
        if q_token in {"2", "0"}:
            return k_token == "1"
        elif q_token in {"1", "3"}:
            return k_token == "0"
        elif q_token in {"4"}:
            return k_token == "5"
        elif q_token in {"5"}:
            return k_token == "4"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_3_pattern = select_closest(tokens, tokens, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_0_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_3_output, attn_1_0_output):
        if attn_1_3_output in {0, 1, 2, 3, 4, 5, 40, 44}:
            return attn_1_0_output == 0
        elif attn_1_3_output in {8, 6, 14}:
            return attn_1_0_output == 45
        elif attn_1_3_output in {32, 33, 7, 18, 20}:
            return attn_1_0_output == 43
        elif attn_1_3_output in {24, 9}:
            return attn_1_0_output == 38
        elif attn_1_3_output in {10}:
            return attn_1_0_output == 16
        elif attn_1_3_output in {26, 11}:
            return attn_1_0_output == 48
        elif attn_1_3_output in {27, 12, 13}:
            return attn_1_0_output == 46
        elif attn_1_3_output in {36, 15}:
            return attn_1_0_output == 49
        elif attn_1_3_output in {16, 39}:
            return attn_1_0_output == 39
        elif attn_1_3_output in {37, 38, 17, 23, 31}:
            return attn_1_0_output == 41
        elif attn_1_3_output in {19, 29}:
            return attn_1_0_output == 47
        elif attn_1_3_output in {34, 21, 22, 25, 30}:
            return attn_1_0_output == 40
        elif attn_1_3_output in {28}:
            return attn_1_0_output == 35
        elif attn_1_3_output in {35}:
            return attn_1_0_output == 44
        elif attn_1_3_output in {41, 42, 49}:
            return attn_1_0_output == 8
        elif attn_1_3_output in {43, 47}:
            return attn_1_0_output == 5
        elif attn_1_3_output in {45}:
            return attn_1_0_output == 7
        elif attn_1_3_output in {48, 46}:
            return attn_1_0_output == 9

    num_attn_2_0_pattern = select(attn_1_0_outputs, attn_1_3_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_0_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_0_1_output, attn_0_2_output):
        if attn_0_1_output in {0, 4}:
            return attn_0_2_output == 0
        elif attn_0_1_output in {1, 35, 9, 14, 24}:
            return attn_0_2_output == 41
        elif attn_0_1_output in {17, 2, 46, 39}:
            return attn_0_2_output == 44
        elif attn_0_1_output in {19, 3, 20, 7}:
            return attn_0_2_output == 46
        elif attn_0_1_output in {32, 37, 5, 48, 25, 26}:
            return attn_0_2_output == 48
        elif attn_0_1_output in {33, 36, 6, 10, 12, 23, 27, 28}:
            return attn_0_2_output == 47
        elif attn_0_1_output in {8, 11, 15, 18, 21, 31}:
            return attn_0_2_output == 43
        elif attn_0_1_output in {41, 45, 13}:
            return attn_0_2_output == 42
        elif attn_0_1_output in {16, 29}:
            return attn_0_2_output == 45
        elif attn_0_1_output in {40, 22}:
            return attn_0_2_output == 40
        elif attn_0_1_output in {34, 38, 43, 47, 49, 30}:
            return attn_0_2_output == 49
        elif attn_0_1_output in {42}:
            return attn_0_2_output == 39
        elif attn_0_1_output in {44}:
            return attn_0_2_output == 10

    num_attn_2_1_pattern = select(attn_0_2_outputs, attn_0_1_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_1_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_0_2_output, attn_0_1_output):
        if attn_0_2_output in {0, 10, 21, 15}:
            return attn_0_1_output == 46
        elif attn_0_2_output in {1, 4, 17}:
            return attn_0_1_output == 35
        elif attn_0_2_output in {33, 2, 35, 8, 9, 26}:
            return attn_0_1_output == 48
        elif attn_0_2_output in {40, 3, 44, 45}:
            return attn_0_1_output == 49
        elif attn_0_2_output in {36, 5}:
            return attn_0_1_output == 38
        elif attn_0_2_output in {46, 34, 6}:
            return attn_0_1_output == 39
        elif attn_0_2_output in {7}:
            return attn_0_1_output == 10
        elif attn_0_2_output in {11}:
            return attn_0_1_output == 30
        elif attn_0_2_output in {12, 29}:
            return attn_0_1_output == 42
        elif attn_0_2_output in {13, 22}:
            return attn_0_1_output == 43
        elif attn_0_2_output in {32, 37, 38, 42, 14, 18, 27}:
            return attn_0_1_output == 44
        elif attn_0_2_output in {16, 24, 48, 43}:
            return attn_0_1_output == 36
        elif attn_0_2_output in {19}:
            return attn_0_1_output == 45
        elif attn_0_2_output in {20}:
            return attn_0_1_output == 16
        elif attn_0_2_output in {25, 41, 23}:
            return attn_0_1_output == 47
        elif attn_0_2_output in {28}:
            return attn_0_1_output == 32
        elif attn_0_2_output in {49, 30}:
            return attn_0_1_output == 41
        elif attn_0_2_output in {31}:
            return attn_0_1_output == 1
        elif attn_0_2_output in {39}:
            return attn_0_1_output == 6
        elif attn_0_2_output in {47}:
            return attn_0_1_output == 40

    num_attn_2_2_pattern = select(attn_0_1_outputs, attn_0_2_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_3_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(token, attn_0_3_output):
        if token in {"<s>", "4", "2", "5", "1", "3", "0"}:
            return attn_0_3_output == ""

    num_attn_2_3_pattern = select(attn_0_3_outputs, tokens, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_1_2_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(mlp_1_1_output, num_mlp_1_1_output):
        key = (mlp_1_1_output, num_mlp_1_1_output)
        if key in {
            (0, 24),
            (1, 24),
            (2, 17),
            (2, 22),
            (2, 24),
            (3, 17),
            (3, 22),
            (3, 24),
            (4, 24),
            (5, 17),
            (5, 22),
            (5, 24),
            (6, 0),
            (6, 1),
            (6, 2),
            (6, 3),
            (6, 4),
            (6, 5),
            (6, 6),
            (6, 7),
            (6, 9),
            (6, 10),
            (6, 11),
            (6, 12),
            (6, 13),
            (6, 14),
            (6, 15),
            (6, 16),
            (6, 17),
            (6, 18),
            (6, 19),
            (6, 20),
            (6, 21),
            (6, 22),
            (6, 23),
            (6, 24),
            (6, 25),
            (6, 26),
            (6, 27),
            (6, 28),
            (6, 29),
            (6, 30),
            (6, 31),
            (6, 32),
            (6, 33),
            (6, 34),
            (6, 35),
            (6, 36),
            (6, 37),
            (6, 38),
            (6, 39),
            (6, 40),
            (6, 41),
            (6, 42),
            (6, 43),
            (6, 44),
            (6, 45),
            (6, 46),
            (6, 47),
            (6, 48),
            (6, 49),
            (7, 24),
            (10, 24),
            (11, 24),
            (12, 17),
            (12, 22),
            (12, 24),
            (13, 24),
            (14, 3),
            (14, 4),
            (14, 5),
            (14, 6),
            (14, 7),
            (14, 10),
            (14, 16),
            (14, 17),
            (14, 18),
            (14, 20),
            (14, 22),
            (14, 23),
            (14, 24),
            (14, 27),
            (14, 28),
            (14, 30),
            (14, 33),
            (14, 34),
            (14, 37),
            (14, 39),
            (14, 41),
            (14, 42),
            (14, 43),
            (14, 48),
            (15, 24),
            (17, 24),
            (18, 22),
            (18, 24),
            (19, 6),
            (19, 17),
            (19, 20),
            (19, 22),
            (19, 24),
            (19, 30),
            (19, 48),
            (20, 24),
            (21, 24),
            (22, 24),
            (23, 22),
            (23, 24),
            (26, 6),
            (26, 17),
            (26, 22),
            (26, 24),
            (27, 22),
            (27, 24),
            (28, 17),
            (28, 22),
            (28, 24),
            (29, 4),
            (29, 5),
            (29, 6),
            (29, 7),
            (29, 10),
            (29, 16),
            (29, 17),
            (29, 18),
            (29, 20),
            (29, 22),
            (29, 23),
            (29, 24),
            (29, 28),
            (29, 30),
            (29, 33),
            (29, 34),
            (29, 37),
            (29, 39),
            (29, 41),
            (29, 42),
            (29, 43),
            (29, 48),
            (30, 17),
            (30, 22),
            (30, 24),
            (31, 17),
            (31, 22),
            (31, 24),
            (34, 24),
            (35, 1),
            (35, 3),
            (35, 4),
            (35, 5),
            (35, 6),
            (35, 7),
            (35, 10),
            (35, 16),
            (35, 17),
            (35, 18),
            (35, 20),
            (35, 22),
            (35, 23),
            (35, 24),
            (35, 27),
            (35, 28),
            (35, 30),
            (35, 32),
            (35, 33),
            (35, 34),
            (35, 37),
            (35, 38),
            (35, 39),
            (35, 40),
            (35, 41),
            (35, 42),
            (35, 43),
            (35, 47),
            (35, 48),
            (35, 49),
            (36, 22),
            (36, 24),
            (37, 24),
            (38, 24),
            (39, 24),
            (40, 24),
            (41, 6),
            (41, 17),
            (41, 20),
            (41, 22),
            (41, 24),
            (41, 30),
            (41, 48),
            (42, 6),
            (42, 17),
            (42, 20),
            (42, 22),
            (42, 24),
            (42, 30),
            (43, 6),
            (43, 17),
            (43, 20),
            (43, 22),
            (43, 24),
            (43, 30),
            (43, 48),
            (44, 24),
            (45, 6),
            (45, 17),
            (45, 18),
            (45, 20),
            (45, 22),
            (45, 24),
            (45, 28),
            (45, 30),
            (45, 33),
            (45, 39),
            (45, 42),
            (45, 48),
            (46, 24),
            (47, 5),
            (47, 6),
            (47, 17),
            (47, 18),
            (47, 20),
            (47, 22),
            (47, 24),
            (47, 28),
            (47, 30),
            (47, 33),
            (47, 39),
            (47, 41),
            (47, 42),
            (47, 48),
            (48, 24),
            (49, 6),
            (49, 17),
            (49, 20),
            (49, 22),
            (49, 24),
            (49, 30),
        }:
            return 39
        elif key in {
            (5, 41),
            (7, 41),
            (10, 41),
            (17, 41),
            (34, 41),
            (38, 41),
            (44, 8),
            (44, 9),
            (44, 11),
            (44, 41),
            (44, 44),
            (48, 41),
        }:
            return 7
        elif key in {(27, 16), (38, 16), (38, 17)}:
            return 40
        return 34

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(mlp_1_1_outputs, num_mlp_1_1_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_1_2_output, attn_1_0_output):
        key = (attn_1_2_output, attn_1_0_output)
        return 44

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_1_2_outputs, attn_1_0_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_0_0_output, num_attn_1_1_output):
        key = (num_attn_0_0_output, num_attn_1_1_output)
        return 32

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_1_output):
        key = num_attn_2_1_output
        return 25

    num_mlp_2_1_outputs = [num_mlp_2_1(k0) for k0 in num_attn_2_1_outputs]
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


print(run(["<s>", "3"]))
