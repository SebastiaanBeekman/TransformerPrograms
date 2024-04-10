import numpy as np
import pandas as pd


def select_closest(keys, queries, predicate):
    scores = [[False for _ in keys] for _ in queries]
    for i, q in enumerate(queries):
        matches = [j for j, k in enumerate(keys[: i + 1]) if predicate(q, k)]
        if not (any(matches)):
            scores[i][0] = True
        else:
            j = min(matches, key=lambda j: len(matches) if j == i else abs(i - j))
            scores[i][j] = True
    return scores


def select(keys, queries, predicate):
    return [[predicate(q, k) for k in keys[: i + 1]] for i, q in enumerate(queries)]


def aggregate(attention, values):
    return [[v for a, v in zip(attn, values) if a][0] for attn in attention]


def aggregate_sum(attention, values):
    return [sum([v for a, v in zip(attn, values) if a]) for attn in attention]


def run(tokens):

    # classifier weights ##########################################
    classifier_weights = pd.read_csv(
        "output/length/rasp/dyck1/trainlength30/s0/dyck1_weights.csv",
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
    def predicate_0_0(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 6
        elif token in {"<s>"}:
            return position == 3

    attn_0_0_pattern = select_closest(positions, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 44
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {8, 2}:
            return k_position == 6
        elif q_position in {41, 3, 52}:
            return k_position == 39
        elif q_position in {4}:
            return k_position == 3
        elif q_position in {33, 5}:
            return k_position == 55
        elif q_position in {6}:
            return k_position == 5
        elif q_position in {55, 36, 13, 7}:
            return k_position == 12
        elif q_position in {9, 50, 20}:
            return k_position == 7
        elif q_position in {10}:
            return k_position == 8
        elif q_position in {11, 12}:
            return k_position == 10
        elif q_position in {35, 14}:
            return k_position == 13
        elif q_position in {16, 15}:
            return k_position == 14
        elif q_position in {17, 46, 39}:
            return k_position == 16
        elif q_position in {18}:
            return k_position == 17
        elif q_position in {19, 21}:
            return k_position == 18
        elif q_position in {24, 56, 22}:
            return k_position == 21
        elif q_position in {23}:
            return k_position == 22
        elif q_position in {25}:
            return k_position == 19
        elif q_position in {26}:
            return k_position == 23
        elif q_position in {27}:
            return k_position == 49
        elif q_position in {28}:
            return k_position == 27
        elif q_position in {29}:
            return k_position == 4
        elif q_position in {43, 30}:
            return k_position == 57
        elif q_position in {34, 37, 31}:
            return k_position == 11
        elif q_position in {32}:
            return k_position == 38
        elif q_position in {38}:
            return k_position == 30
        elif q_position in {40, 51, 53}:
            return k_position == 42
        elif q_position in {42, 59}:
            return k_position == 36
        elif q_position in {44}:
            return k_position == 35
        elif q_position in {45}:
            return k_position == 9
        elif q_position in {47}:
            return k_position == 33
        elif q_position in {48}:
            return k_position == 34
        elif q_position in {49}:
            return k_position == 15
        elif q_position in {54}:
            return k_position == 54
        elif q_position in {57}:
            return k_position == 59
        elif q_position in {58}:
            return k_position == 46

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(token, position):
        if token in {"("}:
            return position == 7
        elif token in {")"}:
            return position == 6
        elif token in {"<s>"}:
            return position == 27

    attn_0_2_pattern = select_closest(positions, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0}:
            return k_position == 59
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 6
        elif q_position in {3, 4}:
            return k_position == 2
        elif q_position in {34, 5, 23}:
            return k_position == 22
        elif q_position in {6}:
            return k_position == 4
        elif q_position in {7}:
            return k_position == 43
        elif q_position in {32, 8, 42, 44, 22}:
            return k_position == 7
        elif q_position in {9}:
            return k_position == 8
        elif q_position in {49, 10}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 10
        elif q_position in {12, 30, 31}:
            return k_position == 11
        elif q_position in {51, 13}:
            return k_position == 12
        elif q_position in {14, 39}:
            return k_position == 13
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {16, 45}:
            return k_position == 15
        elif q_position in {17}:
            return k_position == 16
        elif q_position in {18, 37}:
            return k_position == 17
        elif q_position in {19}:
            return k_position == 18
        elif q_position in {20}:
            return k_position == 19
        elif q_position in {21, 47}:
            return k_position == 20
        elif q_position in {24}:
            return k_position == 23
        elif q_position in {25}:
            return k_position == 24
        elif q_position in {26, 28}:
            return k_position == 25
        elif q_position in {27}:
            return k_position == 26
        elif q_position in {29}:
            return k_position == 29
        elif q_position in {33, 59}:
            return k_position == 51
        elif q_position in {41, 35}:
            return k_position == 55
        elif q_position in {36}:
            return k_position == 45
        elif q_position in {46, 50, 38}:
            return k_position == 38
        elif q_position in {40}:
            return k_position == 36
        elif q_position in {43}:
            return k_position == 44
        elif q_position in {48}:
            return k_position == 54
        elif q_position in {52}:
            return k_position == 37
        elif q_position in {53}:
            return k_position == 42
        elif q_position in {54}:
            return k_position == 33
        elif q_position in {56, 55}:
            return k_position == 47
        elif q_position in {57}:
            return k_position == 28
        elif q_position in {58}:
            return k_position == 41

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # attn_0_4 ####################################################
    def predicate_0_4(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 8
        elif token in {"<s>"}:
            return position == 27

    attn_0_4_pattern = select_closest(positions, tokens, predicate_0_4)
    attn_0_4_outputs = aggregate(attn_0_4_pattern, tokens)
    attn_0_4_output_scores = classifier_weights.loc[
        [("attn_0_4_outputs", str(v)) for v in attn_0_4_outputs]
    ]

    # attn_0_5 ####################################################
    def predicate_0_5(q_position, k_position):
        if q_position in {0}:
            return k_position == 29
        elif q_position in {1}:
            return k_position == 1
        elif q_position in {2}:
            return k_position == 8
        elif q_position in {3, 4, 5, 6, 29}:
            return k_position == 3
        elif q_position in {7}:
            return k_position == 41
        elif q_position in {8, 10, 18, 20, 22, 24}:
            return k_position == 7
        elif q_position in {16, 9, 19, 17}:
            return k_position == 13
        elif q_position in {11}:
            return k_position == 19
        elif q_position in {12}:
            return k_position == 9
        elif q_position in {45, 44, 13}:
            return k_position == 56
        elif q_position in {14, 15}:
            return k_position == 11
        elif q_position in {51, 21}:
            return k_position == 44
        elif q_position in {26, 28, 23}:
            return k_position == 21
        elif q_position in {25, 55}:
            return k_position == 54
        elif q_position in {32, 35, 27}:
            return k_position == 33
        elif q_position in {34, 43, 30}:
            return k_position == 30
        elif q_position in {31}:
            return k_position == 27
        elif q_position in {33, 53}:
            return k_position == 39
        elif q_position in {57, 36, 52}:
            return k_position == 42
        elif q_position in {37, 38}:
            return k_position == 5
        elif q_position in {59, 39}:
            return k_position == 55
        elif q_position in {40, 56}:
            return k_position == 46
        elif q_position in {48, 41}:
            return k_position == 52
        elif q_position in {42}:
            return k_position == 50
        elif q_position in {46}:
            return k_position == 38
        elif q_position in {47}:
            return k_position == 53
        elif q_position in {49}:
            return k_position == 57
        elif q_position in {50, 58}:
            return k_position == 48
        elif q_position in {54}:
            return k_position == 40

    attn_0_5_pattern = select_closest(positions, positions, predicate_0_5)
    attn_0_5_outputs = aggregate(attn_0_5_pattern, tokens)
    attn_0_5_output_scores = classifier_weights.loc[
        [("attn_0_5_outputs", str(v)) for v in attn_0_5_outputs]
    ]

    # attn_0_6 ####################################################
    def predicate_0_6(q_position, k_position):
        if q_position in {0}:
            return k_position == 58
        elif q_position in {1, 36}:
            return k_position == 29
        elif q_position in {2}:
            return k_position == 6
        elif q_position in {3, 4}:
            return k_position == 3
        elif q_position in {50, 5}:
            return k_position == 49
        elif q_position in {29, 6}:
            return k_position == 5
        elif q_position in {17, 19, 7}:
            return k_position == 16
        elif q_position in {8, 9, 20}:
            return k_position == 7
        elif q_position in {10, 28, 14}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 8
        elif q_position in {41, 12}:
            return k_position == 11
        elif q_position in {13}:
            return k_position == 10
        elif q_position in {15}:
            return k_position == 14
        elif q_position in {16}:
            return k_position == 12
        elif q_position in {18, 27, 42}:
            return k_position == 17
        elif q_position in {21, 23}:
            return k_position == 18
        elif q_position in {24, 25, 22}:
            return k_position == 21
        elif q_position in {26, 34}:
            return k_position == 19
        elif q_position in {44, 57, 59, 30, 31}:
            return k_position == 45
        elif q_position in {32}:
            return k_position == 52
        elif q_position in {33}:
            return k_position == 56
        elif q_position in {35}:
            return k_position == 38
        elif q_position in {37, 55}:
            return k_position == 41
        elif q_position in {58, 38}:
            return k_position == 48
        elif q_position in {49, 39}:
            return k_position == 42
        elif q_position in {40, 47}:
            return k_position == 35
        elif q_position in {43}:
            return k_position == 53
        elif q_position in {45}:
            return k_position == 36
        elif q_position in {46}:
            return k_position == 37
        elif q_position in {48}:
            return k_position == 59
        elif q_position in {51}:
            return k_position == 54
        elif q_position in {52}:
            return k_position == 47
        elif q_position in {53}:
            return k_position == 33
        elif q_position in {54}:
            return k_position == 39
        elif q_position in {56}:
            return k_position == 30

    attn_0_6_pattern = select_closest(positions, positions, predicate_0_6)
    attn_0_6_outputs = aggregate(attn_0_6_pattern, tokens)
    attn_0_6_output_scores = classifier_weights.loc[
        [("attn_0_6_outputs", str(v)) for v in attn_0_6_outputs]
    ]

    # attn_0_7 ####################################################
    def predicate_0_7(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 10
        elif token in {"<s>"}:
            return position == 2

    attn_0_7_pattern = select_closest(positions, tokens, predicate_0_7)
    attn_0_7_outputs = aggregate(attn_0_7_pattern, tokens)
    attn_0_7_output_scores = classifier_weights.loc[
        [("attn_0_7_outputs", str(v)) for v in attn_0_7_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {
            0,
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
            50,
            51,
            52,
            54,
            55,
            56,
            57,
            58,
            59,
        }:
            return token == ""
        elif position in {1, 2, 3, 4, 5, 6, 7, 53}:
            return token == ")"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(token, position):
        if token in {"("}:
            return position == 16
        elif token in {")"}:
            return position == 36
        elif token in {"<s>"}:
            return position == 7

    num_attn_0_1_pattern = select(positions, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(token, position):
        if token in {"("}:
            return position == 30
        elif token in {")"}:
            return position == 7
        elif token in {"<s>"}:
            return position == 20

    num_attn_0_2_pattern = select(positions, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(token, position):
        if token in {"("}:
            return position == 26
        elif token in {")"}:
            return position == 36
        elif token in {"<s>"}:
            return position == 3

    num_attn_0_3_pattern = select(positions, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # num_attn_0_4 ####################################################
    def num_predicate_0_4(token, position):
        if token in {"("}:
            return position == 28
        elif token in {")"}:
            return position == 48
        elif token in {"<s>"}:
            return position == 16

    num_attn_0_4_pattern = select(positions, tokens, num_predicate_0_4)
    num_attn_0_4_outputs = aggregate_sum(num_attn_0_4_pattern, ones)
    num_attn_0_4_output_scores = classifier_weights.loc[
        [("num_attn_0_4_outputs", "_") for v in num_attn_0_4_outputs]
    ].mul(num_attn_0_4_outputs, axis=0)

    # num_attn_0_5 ####################################################
    def num_predicate_0_5(token, position):
        if token in {"("}:
            return position == 39
        elif token in {")"}:
            return position == 37
        elif token in {"<s>"}:
            return position == 22

    num_attn_0_5_pattern = select(positions, tokens, num_predicate_0_5)
    num_attn_0_5_outputs = aggregate_sum(num_attn_0_5_pattern, ones)
    num_attn_0_5_output_scores = classifier_weights.loc[
        [("num_attn_0_5_outputs", "_") for v in num_attn_0_5_outputs]
    ].mul(num_attn_0_5_outputs, axis=0)

    # num_attn_0_6 ####################################################
    def num_predicate_0_6(token, position):
        if token in {"("}:
            return position == 9
        elif token in {")"}:
            return position == 36
        elif token in {"<s>"}:
            return position == 41

    num_attn_0_6_pattern = select(positions, tokens, num_predicate_0_6)
    num_attn_0_6_outputs = aggregate_sum(num_attn_0_6_pattern, ones)
    num_attn_0_6_output_scores = classifier_weights.loc[
        [("num_attn_0_6_outputs", "_") for v in num_attn_0_6_outputs]
    ].mul(num_attn_0_6_outputs, axis=0)

    # num_attn_0_7 ####################################################
    def num_predicate_0_7(position, token):
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
            36,
            37,
            38,
            39,
            40,
            41,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
        }:
            return token == ""
        elif position in {42, 35}:
            return token == ")"

    num_attn_0_7_pattern = select(tokens, positions, num_predicate_0_7)
    num_attn_0_7_outputs = aggregate_sum(num_attn_0_7_pattern, ones)
    num_attn_0_7_output_scores = classifier_weights.loc[
        [("num_attn_0_7_outputs", "_") for v in num_attn_0_7_outputs]
    ].mul(num_attn_0_7_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_6_output, attn_0_3_output):
        key = (attn_0_6_output, attn_0_3_output)
        if key in {("(", "("), (")", "(")}:
            return 43
        elif key in {("<s>", "(")}:
            return 24
        elif key in {("(", "<s>"), ("<s>", "<s>")}:
            return 34
        elif key in {(")", ")")}:
            return 2
        return 16

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_6_outputs, attn_0_3_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_6_output, attn_0_4_output):
        key = (attn_0_6_output, attn_0_4_output)
        return 19

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_6_outputs, attn_0_4_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_7_output, num_attn_0_2_output):
        key = (num_attn_0_7_output, num_attn_0_2_output)
        return 36

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_7_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_5_output):
        key = num_attn_0_5_output
        return 19

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_5_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(token, position):
        if token in {"("}:
            return position == 7
        elif token in {")"}:
            return position == 8
        elif token in {"<s>"}:
            return position == 56

    attn_1_0_pattern = select_closest(positions, tokens, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_2_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(token, position):
        if token in {"("}:
            return position == 1
        elif token in {")"}:
            return position == 2
        elif token in {"<s>"}:
            return position == 3

    attn_1_1_pattern = select_closest(positions, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, tokens)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(attn_0_5_output, position):
        if attn_0_5_output in {"("}:
            return position == 3
        elif attn_0_5_output in {")"}:
            return position == 11
        elif attn_0_5_output in {"<s>"}:
            return position == 1

    attn_1_2_pattern = select_closest(positions, attn_0_5_outputs, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_4_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_0_output, attn_0_3_output):
        if attn_0_0_output in {"(", ")", "<s>"}:
            return attn_0_3_output == ")"

    attn_1_3_pattern = select_closest(attn_0_3_outputs, attn_0_0_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_5_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # attn_1_4 ####################################################
    def predicate_1_4(attn_0_2_output, position):
        if attn_0_2_output in {"(", "<s>"}:
            return position == 1
        elif attn_0_2_output in {")"}:
            return position == 9

    attn_1_4_pattern = select_closest(positions, attn_0_2_outputs, predicate_1_4)
    attn_1_4_outputs = aggregate(attn_1_4_pattern, attn_0_4_outputs)
    attn_1_4_output_scores = classifier_weights.loc[
        [("attn_1_4_outputs", str(v)) for v in attn_1_4_outputs]
    ]

    # attn_1_5 ####################################################
    def predicate_1_5(attn_0_4_output, position):
        if attn_0_4_output in {"(", "<s>"}:
            return position == 1
        elif attn_0_4_output in {")"}:
            return position == 16

    attn_1_5_pattern = select_closest(positions, attn_0_4_outputs, predicate_1_5)
    attn_1_5_outputs = aggregate(attn_1_5_pattern, attn_0_0_outputs)
    attn_1_5_output_scores = classifier_weights.loc[
        [("attn_1_5_outputs", str(v)) for v in attn_1_5_outputs]
    ]

    # attn_1_6 ####################################################
    def predicate_1_6(attn_0_0_output, position):
        if attn_0_0_output in {"(", "<s>"}:
            return position == 1
        elif attn_0_0_output in {")"}:
            return position == 11

    attn_1_6_pattern = select_closest(positions, attn_0_0_outputs, predicate_1_6)
    attn_1_6_outputs = aggregate(attn_1_6_pattern, attn_0_7_outputs)
    attn_1_6_output_scores = classifier_weights.loc[
        [("attn_1_6_outputs", str(v)) for v in attn_1_6_outputs]
    ]

    # attn_1_7 ####################################################
    def predicate_1_7(attn_0_2_output, position):
        if attn_0_2_output in {"(", "<s>"}:
            return position == 1
        elif attn_0_2_output in {")"}:
            return position == 13

    attn_1_7_pattern = select_closest(positions, attn_0_2_outputs, predicate_1_7)
    attn_1_7_outputs = aggregate(attn_1_7_pattern, attn_0_4_outputs)
    attn_1_7_output_scores = classifier_weights.loc[
        [("attn_1_7_outputs", str(v)) for v in attn_1_7_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(q_mlp_0_0_output, k_mlp_0_0_output):
        if q_mlp_0_0_output in {
            0,
            1,
            3,
            6,
            8,
            11,
            12,
            13,
            15,
            17,
            18,
            19,
            20,
            22,
            23,
            24,
            25,
            27,
            29,
            30,
            31,
            33,
            34,
            39,
            41,
            42,
            43,
            45,
            46,
            47,
            48,
            49,
            51,
            54,
            56,
            57,
        }:
            return k_mlp_0_0_output == 24
        elif q_mlp_0_0_output in {2}:
            return k_mlp_0_0_output == 37
        elif q_mlp_0_0_output in {4, 7, 40, 16, 21}:
            return k_mlp_0_0_output == 21
        elif q_mlp_0_0_output in {5}:
            return k_mlp_0_0_output == 34
        elif q_mlp_0_0_output in {9, 38}:
            return k_mlp_0_0_output == 58
        elif q_mlp_0_0_output in {10}:
            return k_mlp_0_0_output == 44
        elif q_mlp_0_0_output in {14}:
            return k_mlp_0_0_output == 33
        elif q_mlp_0_0_output in {26}:
            return k_mlp_0_0_output == 46
        elif q_mlp_0_0_output in {28}:
            return k_mlp_0_0_output == 16
        elif q_mlp_0_0_output in {32}:
            return k_mlp_0_0_output == 8
        elif q_mlp_0_0_output in {35}:
            return k_mlp_0_0_output == 25
        elif q_mlp_0_0_output in {36}:
            return k_mlp_0_0_output == 59
        elif q_mlp_0_0_output in {37}:
            return k_mlp_0_0_output == 41
        elif q_mlp_0_0_output in {44}:
            return k_mlp_0_0_output == 9
        elif q_mlp_0_0_output in {50}:
            return k_mlp_0_0_output == 15
        elif q_mlp_0_0_output in {52}:
            return k_mlp_0_0_output == 39
        elif q_mlp_0_0_output in {53}:
            return k_mlp_0_0_output == 31
        elif q_mlp_0_0_output in {55}:
            return k_mlp_0_0_output == 2
        elif q_mlp_0_0_output in {58}:
            return k_mlp_0_0_output == 30
        elif q_mlp_0_0_output in {59}:
            return k_mlp_0_0_output == 43

    num_attn_1_0_pattern = select(mlp_0_0_outputs, mlp_0_0_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_6_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_0_output, num_mlp_0_0_output):
        if attn_0_0_output in {"("}:
            return num_mlp_0_0_output == 42
        elif attn_0_0_output in {")"}:
            return num_mlp_0_0_output == 9
        elif attn_0_0_output in {"<s>"}:
            return num_mlp_0_0_output == 34

    num_attn_1_1_pattern = select(
        num_mlp_0_0_outputs, attn_0_0_outputs, num_predicate_1_1
    )
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_1_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(num_mlp_0_1_output, position):
        if num_mlp_0_1_output in {
            0,
            7,
            10,
            11,
            14,
            15,
            19,
            20,
            24,
            29,
            30,
            32,
            33,
            36,
            37,
            38,
            50,
            52,
            53,
            57,
        }:
            return position == 5
        elif num_mlp_0_1_output in {1, 26}:
            return position == 7
        elif num_mlp_0_1_output in {
            2,
            39,
            40,
            43,
            44,
            45,
            47,
            48,
            17,
            49,
            51,
            21,
            22,
            23,
            25,
            58,
            59,
        }:
            return position == 4
        elif num_mlp_0_1_output in {3, 4, 12, 6}:
            return position == 2
        elif num_mlp_0_1_output in {34, 5, 8, 41, 42, 13, 46, 16, 18, 55, 56, 27}:
            return position == 3
        elif num_mlp_0_1_output in {9, 35, 31}:
            return position == 6
        elif num_mlp_0_1_output in {28}:
            return position == 57
        elif num_mlp_0_1_output in {54}:
            return position == 58

    num_attn_1_2_pattern = select(positions, num_mlp_0_1_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_0_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(num_mlp_0_1_output, attn_0_0_output):
        if num_mlp_0_1_output in {
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            10,
            11,
            12,
            14,
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
            49,
            50,
            51,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
        }:
            return attn_0_0_output == ""
        elif num_mlp_0_1_output in {8, 9, 13, 15, 48}:
            return attn_0_0_output == "("
        elif num_mlp_0_1_output in {52}:
            return attn_0_0_output == "<s>"

    num_attn_1_3_pattern = select(
        attn_0_0_outputs, num_mlp_0_1_outputs, num_predicate_1_3
    )
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_1_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # num_attn_1_4 ####################################################
    def num_predicate_1_4(num_mlp_0_0_output, attn_0_3_output):
        if num_mlp_0_0_output in {
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
            29,
            30,
            31,
            35,
            37,
            38,
            39,
            40,
            41,
            42,
            45,
            46,
            47,
            48,
            49,
            51,
            52,
            54,
            56,
            57,
            58,
            59,
        }:
            return attn_0_3_output == ""
        elif num_mlp_0_0_output in {32, 33, 34, 36, 43, 44, 50, 53, 55, 28}:
            return attn_0_3_output == "("

    num_attn_1_4_pattern = select(
        attn_0_3_outputs, num_mlp_0_0_outputs, num_predicate_1_4
    )
    num_attn_1_4_outputs = aggregate_sum(num_attn_1_4_pattern, num_attn_0_1_outputs)
    num_attn_1_4_output_scores = classifier_weights.loc[
        [("num_attn_1_4_outputs", "_") for v in num_attn_1_4_outputs]
    ].mul(num_attn_1_4_outputs, axis=0)

    # num_attn_1_5 ####################################################
    def num_predicate_1_5(attn_0_3_output, mlp_0_0_output):
        if attn_0_3_output in {"("}:
            return mlp_0_0_output == 17
        elif attn_0_3_output in {")"}:
            return mlp_0_0_output == 24
        elif attn_0_3_output in {"<s>"}:
            return mlp_0_0_output == 29

    num_attn_1_5_pattern = select(mlp_0_0_outputs, attn_0_3_outputs, num_predicate_1_5)
    num_attn_1_5_outputs = aggregate_sum(num_attn_1_5_pattern, num_attn_0_6_outputs)
    num_attn_1_5_output_scores = classifier_weights.loc[
        [("num_attn_1_5_outputs", "_") for v in num_attn_1_5_outputs]
    ].mul(num_attn_1_5_outputs, axis=0)

    # num_attn_1_6 ####################################################
    def num_predicate_1_6(num_mlp_0_1_output, position):
        if num_mlp_0_1_output in {0, 2, 41, 22, 25}:
            return position == 5
        elif num_mlp_0_1_output in {1, 12, 48, 17, 19, 52, 29}:
            return position == 3
        elif num_mlp_0_1_output in {3, 4, 36, 13, 30}:
            return position == 2
        elif num_mlp_0_1_output in {32, 34, 5, 9, 59, 23, 27}:
            return position == 4
        elif num_mlp_0_1_output in {38, 6, 42, 43, 14, 18, 28}:
            return position == 48
        elif num_mlp_0_1_output in {33, 7, 10, 11, 46, 16, 50, 21, 53, 55}:
            return position == 19
        elif num_mlp_0_1_output in {8}:
            return position == 23
        elif num_mlp_0_1_output in {49, 15}:
            return position == 51
        elif num_mlp_0_1_output in {20}:
            return position == 34
        elif num_mlp_0_1_output in {24}:
            return position == 24
        elif num_mlp_0_1_output in {56, 26, 51}:
            return position == 47
        elif num_mlp_0_1_output in {39, 37, 31}:
            return position == 31
        elif num_mlp_0_1_output in {35}:
            return position == 25
        elif num_mlp_0_1_output in {40, 47}:
            return position == 8
        elif num_mlp_0_1_output in {44}:
            return position == 7
        elif num_mlp_0_1_output in {45}:
            return position == 20
        elif num_mlp_0_1_output in {54}:
            return position == 37
        elif num_mlp_0_1_output in {57}:
            return position == 39
        elif num_mlp_0_1_output in {58}:
            return position == 57

    num_attn_1_6_pattern = select(positions, num_mlp_0_1_outputs, num_predicate_1_6)
    num_attn_1_6_outputs = aggregate_sum(num_attn_1_6_pattern, num_attn_0_0_outputs)
    num_attn_1_6_output_scores = classifier_weights.loc[
        [("num_attn_1_6_outputs", "_") for v in num_attn_1_6_outputs]
    ].mul(num_attn_1_6_outputs, axis=0)

    # num_attn_1_7 ####################################################
    def num_predicate_1_7(num_mlp_0_0_output, position):
        if num_mlp_0_0_output in {0, 30}:
            return position == 17
        elif num_mlp_0_0_output in {1, 14, 18, 23, 31}:
            return position == 15
        elif num_mlp_0_0_output in {2, 3, 5, 41, 50}:
            return position == 8
        elif num_mlp_0_0_output in {4}:
            return position == 21
        elif num_mlp_0_0_output in {29, 6}:
            return position == 51
        elif num_mlp_0_0_output in {32, 7, 19, 22, 56}:
            return position == 47
        elif num_mlp_0_0_output in {8, 25, 10, 15}:
            return position == 37
        elif num_mlp_0_0_output in {9, 13}:
            return position == 5
        elif num_mlp_0_0_output in {39, 42, 11, 16, 51, 57}:
            return position == 1
        elif num_mlp_0_0_output in {40, 12}:
            return position == 56
        elif num_mlp_0_0_output in {38, 17, 52, 55, 26, 27}:
            return position == 19
        elif num_mlp_0_0_output in {35, 37, 43, 45, 48, 49, 20, 21, 58}:
            return position == 48
        elif num_mlp_0_0_output in {24, 54, 28, 46}:
            return position == 27
        elif num_mlp_0_0_output in {33}:
            return position == 50
        elif num_mlp_0_0_output in {34, 47}:
            return position == 22
        elif num_mlp_0_0_output in {36}:
            return position == 4
        elif num_mlp_0_0_output in {44}:
            return position == 7
        elif num_mlp_0_0_output in {53}:
            return position == 32
        elif num_mlp_0_0_output in {59}:
            return position == 57

    num_attn_1_7_pattern = select(positions, num_mlp_0_0_outputs, num_predicate_1_7)
    num_attn_1_7_outputs = aggregate_sum(num_attn_1_7_pattern, num_attn_0_1_outputs)
    num_attn_1_7_output_scores = classifier_weights.loc[
        [("num_attn_1_7_outputs", "_") for v in num_attn_1_7_outputs]
    ].mul(num_attn_1_7_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_3_output, attn_1_0_output):
        key = (attn_1_3_output, attn_1_0_output)
        if key in {("(", "("), ("<s>", "(")}:
            return 40
        elif key in {(")", "<s>"), ("<s>", "<s>")}:
            return 2
        return 13

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_3_outputs, attn_1_0_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_0_0_output, attn_0_7_output):
        key = (attn_0_0_output, attn_0_7_output)
        if key in {("<s>", ")"), ("<s>", "<s>")}:
            return 22
        return 28

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_0_7_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_2_output, num_attn_1_6_output):
        key = (num_attn_1_2_output, num_attn_1_6_output)
        if key in {
            (15, 0),
            (16, 0),
            (17, 0),
            (18, 0),
            (19, 0),
            (20, 0),
            (21, 0),
            (22, 0),
            (22, 1),
            (23, 0),
            (23, 1),
            (24, 0),
            (24, 1),
            (25, 0),
            (25, 1),
            (26, 0),
            (26, 1),
            (27, 0),
            (27, 1),
            (28, 0),
            (28, 1),
            (29, 0),
            (29, 1),
            (30, 0),
            (30, 1),
            (30, 2),
            (31, 0),
            (31, 1),
            (31, 2),
            (32, 0),
            (32, 1),
            (32, 2),
            (33, 0),
            (33, 1),
            (33, 2),
            (34, 0),
            (34, 1),
            (34, 2),
            (35, 0),
            (35, 1),
            (35, 2),
            (36, 0),
            (36, 1),
            (36, 2),
            (37, 0),
            (37, 1),
            (37, 2),
            (38, 0),
            (38, 1),
            (38, 2),
            (38, 3),
            (39, 0),
            (39, 1),
            (39, 2),
            (39, 3),
            (40, 0),
            (40, 1),
            (40, 2),
            (40, 3),
            (41, 0),
            (41, 1),
            (41, 2),
            (41, 3),
            (42, 0),
            (42, 1),
            (42, 2),
            (42, 3),
            (43, 0),
            (43, 1),
            (43, 2),
            (43, 3),
            (44, 0),
            (44, 1),
            (44, 2),
            (44, 3),
            (45, 0),
            (45, 1),
            (45, 2),
            (45, 3),
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
            (49, 0),
            (49, 1),
            (49, 2),
            (49, 3),
            (49, 4),
            (50, 0),
            (50, 1),
            (50, 2),
            (50, 3),
            (50, 4),
            (51, 0),
            (51, 1),
            (51, 2),
            (51, 3),
            (51, 4),
            (52, 0),
            (52, 1),
            (52, 2),
            (52, 3),
            (52, 4),
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
            (60, 0),
            (60, 1),
            (60, 2),
            (60, 3),
            (60, 4),
            (60, 5),
            (61, 0),
            (61, 1),
            (61, 2),
            (61, 3),
            (61, 4),
            (61, 5),
            (61, 6),
            (62, 0),
            (62, 1),
            (62, 2),
            (62, 3),
            (62, 4),
            (62, 5),
            (62, 6),
            (63, 0),
            (63, 1),
            (63, 2),
            (63, 3),
            (63, 4),
            (63, 5),
            (63, 6),
            (64, 0),
            (64, 1),
            (64, 2),
            (64, 3),
            (64, 4),
            (64, 5),
            (64, 6),
            (65, 0),
            (65, 1),
            (65, 2),
            (65, 3),
            (65, 4),
            (65, 5),
            (65, 6),
            (66, 0),
            (66, 1),
            (66, 2),
            (66, 3),
            (66, 4),
            (66, 5),
            (66, 6),
            (67, 0),
            (67, 1),
            (67, 2),
            (67, 3),
            (67, 4),
            (67, 5),
            (67, 6),
            (68, 0),
            (68, 1),
            (68, 2),
            (68, 3),
            (68, 4),
            (68, 5),
            (68, 6),
            (69, 0),
            (69, 1),
            (69, 2),
            (69, 3),
            (69, 4),
            (69, 5),
            (69, 6),
            (69, 7),
            (70, 0),
            (70, 1),
            (70, 2),
            (70, 3),
            (70, 4),
            (70, 5),
            (70, 6),
            (70, 7),
            (71, 0),
            (71, 1),
            (71, 2),
            (71, 3),
            (71, 4),
            (71, 5),
            (71, 6),
            (71, 7),
            (72, 0),
            (72, 1),
            (72, 2),
            (72, 3),
            (72, 4),
            (72, 5),
            (72, 6),
            (72, 7),
            (73, 0),
            (73, 1),
            (73, 2),
            (73, 3),
            (73, 4),
            (73, 5),
            (73, 6),
            (73, 7),
            (74, 0),
            (74, 1),
            (74, 2),
            (74, 3),
            (74, 4),
            (74, 5),
            (74, 6),
            (74, 7),
            (75, 0),
            (75, 1),
            (75, 2),
            (75, 3),
            (75, 4),
            (75, 5),
            (75, 6),
            (75, 7),
            (76, 0),
            (76, 1),
            (76, 2),
            (76, 3),
            (76, 4),
            (76, 5),
            (76, 6),
            (76, 7),
            (76, 8),
            (77, 0),
            (77, 1),
            (77, 2),
            (77, 3),
            (77, 4),
            (77, 5),
            (77, 6),
            (77, 7),
            (77, 8),
            (78, 0),
            (78, 1),
            (78, 2),
            (78, 3),
            (78, 4),
            (78, 5),
            (78, 6),
            (78, 7),
            (78, 8),
            (79, 0),
            (79, 1),
            (79, 2),
            (79, 3),
            (79, 4),
            (79, 5),
            (79, 6),
            (79, 7),
            (79, 8),
            (80, 0),
            (80, 1),
            (80, 2),
            (80, 3),
            (80, 4),
            (80, 5),
            (80, 6),
            (80, 7),
            (80, 8),
            (81, 0),
            (81, 1),
            (81, 2),
            (81, 3),
            (81, 4),
            (81, 5),
            (81, 6),
            (81, 7),
            (81, 8),
            (82, 0),
            (82, 1),
            (82, 2),
            (82, 3),
            (82, 4),
            (82, 5),
            (82, 6),
            (82, 7),
            (82, 8),
            (83, 0),
            (83, 1),
            (83, 2),
            (83, 3),
            (83, 4),
            (83, 5),
            (83, 6),
            (83, 7),
            (83, 8),
            (84, 0),
            (84, 1),
            (84, 2),
            (84, 3),
            (84, 4),
            (84, 5),
            (84, 6),
            (84, 7),
            (84, 8),
            (84, 9),
            (85, 0),
            (85, 1),
            (85, 2),
            (85, 3),
            (85, 4),
            (85, 5),
            (85, 6),
            (85, 7),
            (85, 8),
            (85, 9),
            (86, 0),
            (86, 1),
            (86, 2),
            (86, 3),
            (86, 4),
            (86, 5),
            (86, 6),
            (86, 7),
            (86, 8),
            (86, 9),
            (87, 0),
            (87, 1),
            (87, 2),
            (87, 3),
            (87, 4),
            (87, 5),
            (87, 6),
            (87, 7),
            (87, 8),
            (87, 9),
            (88, 0),
            (88, 1),
            (88, 2),
            (88, 3),
            (88, 4),
            (88, 5),
            (88, 6),
            (88, 7),
            (88, 8),
            (88, 9),
            (89, 0),
            (89, 1),
            (89, 2),
            (89, 3),
            (89, 4),
            (89, 5),
            (89, 6),
            (89, 7),
            (89, 8),
            (89, 9),
            (90, 0),
            (90, 1),
            (90, 2),
            (90, 3),
            (90, 4),
            (90, 5),
            (90, 6),
            (90, 7),
            (90, 8),
            (90, 9),
            (91, 0),
            (91, 1),
            (91, 2),
            (91, 3),
            (91, 4),
            (91, 5),
            (91, 6),
            (91, 7),
            (91, 8),
            (91, 9),
            (92, 0),
            (92, 1),
            (92, 2),
            (92, 3),
            (92, 4),
            (92, 5),
            (92, 6),
            (92, 7),
            (92, 8),
            (92, 9),
            (92, 10),
            (93, 0),
            (93, 1),
            (93, 2),
            (93, 3),
            (93, 4),
            (93, 5),
            (93, 6),
            (93, 7),
            (93, 8),
            (93, 9),
            (93, 10),
            (94, 0),
            (94, 1),
            (94, 2),
            (94, 3),
            (94, 4),
            (94, 5),
            (94, 6),
            (94, 7),
            (94, 8),
            (94, 9),
            (94, 10),
            (95, 0),
            (95, 1),
            (95, 2),
            (95, 3),
            (95, 4),
            (95, 5),
            (95, 6),
            (95, 7),
            (95, 8),
            (95, 9),
            (95, 10),
            (96, 0),
            (96, 1),
            (96, 2),
            (96, 3),
            (96, 4),
            (96, 5),
            (96, 6),
            (96, 7),
            (96, 8),
            (96, 9),
            (96, 10),
            (97, 0),
            (97, 1),
            (97, 2),
            (97, 3),
            (97, 4),
            (97, 5),
            (97, 6),
            (97, 7),
            (97, 8),
            (97, 9),
            (97, 10),
            (98, 0),
            (98, 1),
            (98, 2),
            (98, 3),
            (98, 4),
            (98, 5),
            (98, 6),
            (98, 7),
            (98, 8),
            (98, 9),
            (98, 10),
            (99, 0),
            (99, 1),
            (99, 2),
            (99, 3),
            (99, 4),
            (99, 5),
            (99, 6),
            (99, 7),
            (99, 8),
            (99, 9),
            (99, 10),
            (99, 11),
            (100, 0),
            (100, 1),
            (100, 2),
            (100, 3),
            (100, 4),
            (100, 5),
            (100, 6),
            (100, 7),
            (100, 8),
            (100, 9),
            (100, 10),
            (100, 11),
            (101, 0),
            (101, 1),
            (101, 2),
            (101, 3),
            (101, 4),
            (101, 5),
            (101, 6),
            (101, 7),
            (101, 8),
            (101, 9),
            (101, 10),
            (101, 11),
            (102, 0),
            (102, 1),
            (102, 2),
            (102, 3),
            (102, 4),
            (102, 5),
            (102, 6),
            (102, 7),
            (102, 8),
            (102, 9),
            (102, 10),
            (102, 11),
            (103, 0),
            (103, 1),
            (103, 2),
            (103, 3),
            (103, 4),
            (103, 5),
            (103, 6),
            (103, 7),
            (103, 8),
            (103, 9),
            (103, 10),
            (103, 11),
            (104, 0),
            (104, 1),
            (104, 2),
            (104, 3),
            (104, 4),
            (104, 5),
            (104, 6),
            (104, 7),
            (104, 8),
            (104, 9),
            (104, 10),
            (104, 11),
            (105, 0),
            (105, 1),
            (105, 2),
            (105, 3),
            (105, 4),
            (105, 5),
            (105, 6),
            (105, 7),
            (105, 8),
            (105, 9),
            (105, 10),
            (105, 11),
            (106, 0),
            (106, 1),
            (106, 2),
            (106, 3),
            (106, 4),
            (106, 5),
            (106, 6),
            (106, 7),
            (106, 8),
            (106, 9),
            (106, 10),
            (106, 11),
            (107, 0),
            (107, 1),
            (107, 2),
            (107, 3),
            (107, 4),
            (107, 5),
            (107, 6),
            (107, 7),
            (107, 8),
            (107, 9),
            (107, 10),
            (107, 11),
            (107, 12),
            (108, 0),
            (108, 1),
            (108, 2),
            (108, 3),
            (108, 4),
            (108, 5),
            (108, 6),
            (108, 7),
            (108, 8),
            (108, 9),
            (108, 10),
            (108, 11),
            (108, 12),
            (109, 0),
            (109, 1),
            (109, 2),
            (109, 3),
            (109, 4),
            (109, 5),
            (109, 6),
            (109, 7),
            (109, 8),
            (109, 9),
            (109, 10),
            (109, 11),
            (109, 12),
            (110, 0),
            (110, 1),
            (110, 2),
            (110, 3),
            (110, 4),
            (110, 5),
            (110, 6),
            (110, 7),
            (110, 8),
            (110, 9),
            (110, 10),
            (110, 11),
            (110, 12),
            (111, 0),
            (111, 1),
            (111, 2),
            (111, 3),
            (111, 4),
            (111, 5),
            (111, 6),
            (111, 7),
            (111, 8),
            (111, 9),
            (111, 10),
            (111, 11),
            (111, 12),
            (112, 0),
            (112, 1),
            (112, 2),
            (112, 3),
            (112, 4),
            (112, 5),
            (112, 6),
            (112, 7),
            (112, 8),
            (112, 9),
            (112, 10),
            (112, 11),
            (112, 12),
            (113, 0),
            (113, 1),
            (113, 2),
            (113, 3),
            (113, 4),
            (113, 5),
            (113, 6),
            (113, 7),
            (113, 8),
            (113, 9),
            (113, 10),
            (113, 11),
            (113, 12),
            (114, 0),
            (114, 1),
            (114, 2),
            (114, 3),
            (114, 4),
            (114, 5),
            (114, 6),
            (114, 7),
            (114, 8),
            (114, 9),
            (114, 10),
            (114, 11),
            (114, 12),
            (115, 0),
            (115, 1),
            (115, 2),
            (115, 3),
            (115, 4),
            (115, 5),
            (115, 6),
            (115, 7),
            (115, 8),
            (115, 9),
            (115, 10),
            (115, 11),
            (115, 12),
            (115, 13),
            (116, 0),
            (116, 1),
            (116, 2),
            (116, 3),
            (116, 4),
            (116, 5),
            (116, 6),
            (116, 7),
            (116, 8),
            (116, 9),
            (116, 10),
            (116, 11),
            (116, 12),
            (116, 13),
            (117, 0),
            (117, 1),
            (117, 2),
            (117, 3),
            (117, 4),
            (117, 5),
            (117, 6),
            (117, 7),
            (117, 8),
            (117, 9),
            (117, 10),
            (117, 11),
            (117, 12),
            (117, 13),
            (118, 0),
            (118, 1),
            (118, 2),
            (118, 3),
            (118, 4),
            (118, 5),
            (118, 6),
            (118, 7),
            (118, 8),
            (118, 9),
            (118, 10),
            (118, 11),
            (118, 12),
            (118, 13),
            (119, 0),
            (119, 1),
            (119, 2),
            (119, 3),
            (119, 4),
            (119, 5),
            (119, 6),
            (119, 7),
            (119, 8),
            (119, 9),
            (119, 10),
            (119, 11),
            (119, 12),
            (119, 13),
        }:
            return 1
        return 21

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_1_6_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_3_output, num_attn_1_2_output):
        key = (num_attn_1_3_output, num_attn_1_2_output)
        return 11

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_1_2_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(attn_0_0_output, position):
        if attn_0_0_output in {"("}:
            return position == 2
        elif attn_0_0_output in {")"}:
            return position == 5
        elif attn_0_0_output in {"<s>"}:
            return position == 1

    attn_2_0_pattern = select_closest(positions, attn_0_0_outputs, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_7_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(attn_0_4_output, position):
        if attn_0_4_output in {"("}:
            return position == 3
        elif attn_0_4_output in {")"}:
            return position == 11
        elif attn_0_4_output in {"<s>"}:
            return position == 4

    attn_2_1_pattern = select_closest(positions, attn_0_4_outputs, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_1_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(token, mlp_0_0_output):
        if token in {"("}:
            return mlp_0_0_output == 47
        elif token in {")", "<s>"}:
            return mlp_0_0_output == 2

    attn_2_2_pattern = select_closest(mlp_0_0_outputs, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_0_5_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(attn_0_2_output, mlp_0_0_output):
        if attn_0_2_output in {"(", ")"}:
            return mlp_0_0_output == 2
        elif attn_0_2_output in {"<s>"}:
            return mlp_0_0_output == 39

    attn_2_3_pattern = select_closest(mlp_0_0_outputs, attn_0_2_outputs, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_4_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # attn_2_4 ####################################################
    def predicate_2_4(token, position):
        if token in {"(", ")"}:
            return position == 3
        elif token in {"<s>"}:
            return position == 5

    attn_2_4_pattern = select_closest(positions, tokens, predicate_2_4)
    attn_2_4_outputs = aggregate(attn_2_4_pattern, attn_1_1_outputs)
    attn_2_4_output_scores = classifier_weights.loc[
        [("attn_2_4_outputs", str(v)) for v in attn_2_4_outputs]
    ]

    # attn_2_5 ####################################################
    def predicate_2_5(attn_0_3_output, position):
        if attn_0_3_output in {"("}:
            return position == 3
        elif attn_0_3_output in {")"}:
            return position == 4
        elif attn_0_3_output in {"<s>"}:
            return position == 2

    attn_2_5_pattern = select_closest(positions, attn_0_3_outputs, predicate_2_5)
    attn_2_5_outputs = aggregate(attn_2_5_pattern, attn_1_6_outputs)
    attn_2_5_output_scores = classifier_weights.loc[
        [("attn_2_5_outputs", str(v)) for v in attn_2_5_outputs]
    ]

    # attn_2_6 ####################################################
    def predicate_2_6(token, position):
        if token in {"("}:
            return position == 3
        elif token in {")"}:
            return position == 9
        elif token in {"<s>"}:
            return position == 5

    attn_2_6_pattern = select_closest(positions, tokens, predicate_2_6)
    attn_2_6_outputs = aggregate(attn_2_6_pattern, attn_1_0_outputs)
    attn_2_6_output_scores = classifier_weights.loc[
        [("attn_2_6_outputs", str(v)) for v in attn_2_6_outputs]
    ]

    # attn_2_7 ####################################################
    def predicate_2_7(token, position):
        if token in {"(", ")"}:
            return position == 9
        elif token in {"<s>"}:
            return position == 5

    attn_2_7_pattern = select_closest(positions, tokens, predicate_2_7)
    attn_2_7_outputs = aggregate(attn_2_7_pattern, attn_0_0_outputs)
    attn_2_7_output_scores = classifier_weights.loc[
        [("attn_2_7_outputs", str(v)) for v in attn_2_7_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_5_output, attn_0_0_output):
        if attn_1_5_output in {"(", "<s>"}:
            return attn_0_0_output == ")"
        elif attn_1_5_output in {")"}:
            return attn_0_0_output == ""

    num_attn_2_0_pattern = select(attn_0_0_outputs, attn_1_5_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_0_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_5_output, token):
        if attn_1_5_output in {"(", ")", "<s>"}:
            return token == ""

    num_attn_2_1_pattern = select(tokens, attn_1_5_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_1_1_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_5_output, mlp_0_0_output):
        if attn_1_5_output in {"("}:
            return mlp_0_0_output == 16
        elif attn_1_5_output in {")"}:
            return mlp_0_0_output == 58
        elif attn_1_5_output in {"<s>"}:
            return mlp_0_0_output == 2

    num_attn_2_2_pattern = select(mlp_0_0_outputs, attn_1_5_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_2_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(mlp_0_1_output, token):
        if mlp_0_1_output in {0, 41, 17, 18, 30, 31}:
            return token == "("
        elif mlp_0_1_output in {
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
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            42,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            52,
            55,
            56,
            57,
            59,
        }:
            return token == ""
        elif mlp_0_1_output in {58, 43, 54}:
            return token == "<pad>"
        elif mlp_0_1_output in {51, 53}:
            return token == "<s>"

    num_attn_2_3_pattern = select(tokens, mlp_0_1_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_5_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # num_attn_2_4 ####################################################
    def num_predicate_2_4(num_mlp_0_0_output, mlp_0_0_output):
        if num_mlp_0_0_output in {0}:
            return mlp_0_0_output == 56
        elif num_mlp_0_0_output in {
            1,
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
            20,
            21,
            22,
            24,
            25,
            26,
            28,
            29,
            30,
            31,
            32,
            34,
            36,
            37,
            38,
            39,
            40,
            41,
            43,
            45,
            46,
            47,
            49,
            50,
            51,
            52,
            53,
            55,
            56,
            57,
            59,
        }:
            return mlp_0_0_output == 2
        elif num_mlp_0_0_output in {33, 2, 44, 17, 27}:
            return mlp_0_0_output == 16
        elif num_mlp_0_0_output in {3}:
            return mlp_0_0_output == 44
        elif num_mlp_0_0_output in {48, 42, 4}:
            return mlp_0_0_output == 59
        elif num_mlp_0_0_output in {16}:
            return mlp_0_0_output == 14
        elif num_mlp_0_0_output in {18}:
            return mlp_0_0_output == 24
        elif num_mlp_0_0_output in {35, 58, 19}:
            return mlp_0_0_output == 35
        elif num_mlp_0_0_output in {23}:
            return mlp_0_0_output == 17
        elif num_mlp_0_0_output in {54}:
            return mlp_0_0_output == 58

    num_attn_2_4_pattern = select(
        mlp_0_0_outputs, num_mlp_0_0_outputs, num_predicate_2_4
    )
    num_attn_2_4_outputs = aggregate_sum(num_attn_2_4_pattern, num_attn_0_0_outputs)
    num_attn_2_4_output_scores = classifier_weights.loc[
        [("num_attn_2_4_outputs", "_") for v in num_attn_2_4_outputs]
    ].mul(num_attn_2_4_outputs, axis=0)

    # num_attn_2_5 ####################################################
    def num_predicate_2_5(attn_1_5_output, mlp_0_0_output):
        if attn_1_5_output in {"(", ")", "<s>"}:
            return mlp_0_0_output == 2

    num_attn_2_5_pattern = select(mlp_0_0_outputs, attn_1_5_outputs, num_predicate_2_5)
    num_attn_2_5_outputs = aggregate_sum(num_attn_2_5_pattern, num_attn_0_2_outputs)
    num_attn_2_5_output_scores = classifier_weights.loc[
        [("num_attn_2_5_outputs", "_") for v in num_attn_2_5_outputs]
    ].mul(num_attn_2_5_outputs, axis=0)

    # num_attn_2_6 ####################################################
    def num_predicate_2_6(attn_1_3_output, attn_1_5_output):
        if attn_1_3_output in {"(", ")", "<s>"}:
            return attn_1_5_output == ""

    num_attn_2_6_pattern = select(attn_1_5_outputs, attn_1_3_outputs, num_predicate_2_6)
    num_attn_2_6_outputs = aggregate_sum(num_attn_2_6_pattern, num_attn_0_7_outputs)
    num_attn_2_6_output_scores = classifier_weights.loc[
        [("num_attn_2_6_outputs", "_") for v in num_attn_2_6_outputs]
    ].mul(num_attn_2_6_outputs, axis=0)

    # num_attn_2_7 ####################################################
    def num_predicate_2_7(position, token):
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
            50,
            51,
            52,
            54,
            55,
            56,
            57,
            58,
            59,
        }:
            return token == ""
        elif position in {53, 37}:
            return token == ")"

    num_attn_2_7_pattern = select(tokens, positions, num_predicate_2_7)
    num_attn_2_7_outputs = aggregate_sum(num_attn_2_7_pattern, num_attn_0_3_outputs)
    num_attn_2_7_output_scores = classifier_weights.loc[
        [("num_attn_2_7_outputs", "_") for v in num_attn_2_7_outputs]
    ].mul(num_attn_2_7_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_3_output):
        key = attn_2_3_output
        if key in {""}:
            return 3
        elif key in {""}:
            return 4
        elif key in {")"}:
            return 10
        elif key in {""}:
            return 56
        return 41

    mlp_2_0_outputs = [mlp_2_0(k0) for k0 in attn_2_3_outputs]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_2_5_output, attn_2_0_output):
        key = (attn_2_5_output, attn_2_0_output)
        if key in {(")", ")"), (")", "<s>")}:
            return 47
        elif key in {(")", "(")}:
            return 21
        return 8

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_2_5_outputs, attn_2_0_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_7_output, num_attn_1_4_output):
        key = (num_attn_1_7_output, num_attn_1_4_output)
        return 46

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_7_outputs, num_attn_1_4_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_7_output, num_attn_1_6_output):
        key = (num_attn_2_7_output, num_attn_1_6_output)
        return 46

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_7_outputs, num_attn_1_6_outputs)
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
                attn_0_4_output_scores,
                attn_0_5_output_scores,
                attn_0_6_output_scores,
                attn_0_7_output_scores,
                mlp_0_0_output_scores,
                mlp_0_1_output_scores,
                num_mlp_0_0_output_scores,
                num_mlp_0_1_output_scores,
                attn_1_0_output_scores,
                attn_1_1_output_scores,
                attn_1_2_output_scores,
                attn_1_3_output_scores,
                attn_1_4_output_scores,
                attn_1_5_output_scores,
                attn_1_6_output_scores,
                attn_1_7_output_scores,
                mlp_1_0_output_scores,
                mlp_1_1_output_scores,
                num_mlp_1_0_output_scores,
                num_mlp_1_1_output_scores,
                attn_2_0_output_scores,
                attn_2_1_output_scores,
                attn_2_2_output_scores,
                attn_2_3_output_scores,
                attn_2_4_output_scores,
                attn_2_5_output_scores,
                attn_2_6_output_scores,
                attn_2_7_output_scores,
                mlp_2_0_output_scores,
                mlp_2_1_output_scores,
                num_mlp_2_0_output_scores,
                num_mlp_2_1_output_scores,
                one_scores,
                num_attn_0_0_output_scores,
                num_attn_0_1_output_scores,
                num_attn_0_2_output_scores,
                num_attn_0_3_output_scores,
                num_attn_0_4_output_scores,
                num_attn_0_5_output_scores,
                num_attn_0_6_output_scores,
                num_attn_0_7_output_scores,
                num_attn_1_0_output_scores,
                num_attn_1_1_output_scores,
                num_attn_1_2_output_scores,
                num_attn_1_3_output_scores,
                num_attn_1_4_output_scores,
                num_attn_1_5_output_scores,
                num_attn_1_6_output_scores,
                num_attn_1_7_output_scores,
                num_attn_2_0_output_scores,
                num_attn_2_1_output_scores,
                num_attn_2_2_output_scores,
                num_attn_2_3_output_scores,
                num_attn_2_4_output_scores,
                num_attn_2_5_output_scores,
                num_attn_2_6_output_scores,
                num_attn_2_7_output_scores,
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
            ")",
            ")",
            "(",
            ")",
            ")",
            ")",
            ")",
            ")",
            ")",
            ")",
            "(",
            "(",
            ")",
            "(",
            "(",
            "(",
            "(",
            "(",
            ")",
            "(",
            ")",
            ")",
            "(",
            "(",
            ")",
            ")",
            ")",
            ")",
            "(",
        ]
    )
)
