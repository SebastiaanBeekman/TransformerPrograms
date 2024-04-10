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
        "output/length/rasp/double_hist/trainlength40/s1/double_hist_weights.csv",
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
        if q_token in {"0", "4", "3"}:
            return k_token == "3"
        elif q_token in {"5", "1"}:
            return k_token == "1"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"<s>"}:
            return k_token == "4"

    attn_0_0_pattern = select_closest(tokens, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, tokens)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1"}:
            return k_token == "1"
        elif q_token in {"2", "5"}:
            return k_token == "5"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_0_1_pattern = select_closest(tokens, tokens, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_token, k_token):
        if q_token in {"0", "4"}:
            return k_token == "4"
        elif q_token in {"2", "1"}:
            return k_token == "5"
        elif q_token in {"5", "3"}:
            return k_token == "3"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_0_2_pattern = select_closest(tokens, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 40, 41, 43, 44, 49}:
            return k_position == 4
        elif q_position in {1}:
            return k_position == 5
        elif q_position in {2, 4, 6, 7}:
            return k_position == 7
        elif q_position in {3}:
            return k_position == 22
        elif q_position in {42, 45, 5}:
            return k_position == 6
        elif q_position in {8}:
            return k_position == 1
        elif q_position in {9, 18, 26, 12}:
            return k_position == 8
        elif q_position in {33, 39, 10, 13, 20, 22, 31}:
            return k_position == 10
        elif q_position in {11}:
            return k_position == 2
        elif q_position in {14, 21, 23, 28, 30}:
            return k_position == 11
        elif q_position in {16, 24, 29, 15}:
            return k_position == 9
        elif q_position in {48, 17, 25}:
            return k_position == 17
        elif q_position in {27, 34, 19}:
            return k_position == 16
        elif q_position in {32}:
            return k_position == 20
        elif q_position in {35}:
            return k_position == 21
        elif q_position in {36, 47}:
            return k_position == 12
        elif q_position in {37, 38}:
            return k_position == 15
        elif q_position in {46}:
            return k_position == 26

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, positions)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(position, token):
        if position in {
            0,
            1,
            2,
            3,
            4,
            5,
            6,
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
        elif position in {7}:
            return token == "<s>"

    num_attn_0_0_pattern = select(tokens, positions, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(token, position):
        if token in {"0"}:
            return position == 49
        elif token in {"1"}:
            return position == 39
        elif token in {"2"}:
            return position == 32
        elif token in {"3"}:
            return position == 40
        elif token in {"4", "5"}:
            return position == 33
        elif token in {"<s>"}:
            return position == 46

    num_attn_0_1_pattern = select(positions, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_token, k_token):
        if q_token in {"3", "0", "1", "4", "<s>"}:
            return k_token == ""
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"5"}:
            return k_token == "5"

    num_attn_0_2_pattern = select(tokens, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_token, k_token):
        if q_token in {"3", "0", "1", "5", "2", "4", "<s>"}:
            return k_token == ""

    num_attn_0_3_pattern = select(tokens, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(attn_0_2_output, attn_0_0_output):
        key = (attn_0_2_output, attn_0_0_output)
        if key in {("3", "3"), ("3", "<s>")}:
            return 39
        return 7

    mlp_0_0_outputs = [
        mlp_0_0(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_0_outputs)
    ]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_2_output, attn_0_0_output):
        key = (attn_0_2_output, attn_0_0_output)
        if key in {
            ("0", "0"),
            ("0", "3"),
            ("2", "0"),
            ("3", "0"),
            ("4", "0"),
            ("4", "1"),
            ("4", "2"),
            ("4", "3"),
            ("5", "0"),
            ("5", "1"),
            ("5", "2"),
            ("5", "5"),
        }:
            return 2
        elif key in {("2", "3")}:
            return 1
        elif key in {("3", "3")}:
            return 39
        return 3

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_0_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output, num_attn_0_2_output):
        key = (num_attn_0_0_output, num_attn_0_2_output)
        return 24

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_2_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_0_output, num_attn_0_3_output):
        key = (num_attn_0_0_output, num_attn_0_3_output)
        return 31

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(token, position):
        if token in {"0", "1", "5"}:
            return position == 5
        elif token in {"2"}:
            return position == 8
        elif token in {"<s>", "3"}:
            return position == 10
        elif token in {"4"}:
            return position == 12

    attn_1_0_pattern = select_closest(positions, tokens, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_3_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_token, k_token):
        if q_token in {"0", "5", "4"}:
            return k_token == "2"
        elif q_token in {"2", "1", "3"}:
            return k_token == "0"
        elif q_token in {"<s>"}:
            return k_token == "3"

    attn_1_1_pattern = select_closest(tokens, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, mlp_0_1_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(q_token, k_token):
        if q_token in {"2", "0", "1", "5"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_1_2_pattern = select_closest(tokens, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, mlp_0_1_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(q_position, k_position):
        if q_position in {0, 13}:
            return k_position == 22
        elif q_position in {1, 46, 31}:
            return k_position == 4
        elif q_position in {35, 2, 19}:
            return k_position == 25
        elif q_position in {26, 3, 44}:
            return k_position == 2
        elif q_position in {40, 4, 45}:
            return k_position == 12
        elif q_position in {36, 5}:
            return k_position == 21
        elif q_position in {6, 15}:
            return k_position == 32
        elif q_position in {7, 10, 11, 17, 22}:
            return k_position == 16
        elif q_position in {8, 42, 12}:
            return k_position == 13
        elif q_position in {9}:
            return k_position == 26
        elif q_position in {14}:
            return k_position == 10
        elif q_position in {16}:
            return k_position == 5
        elif q_position in {18, 37}:
            return k_position == 38
        elif q_position in {34, 20, 29, 39}:
            return k_position == 35
        elif q_position in {49, 21}:
            return k_position == 17
        elif q_position in {47, 30, 23}:
            return k_position == 8
        elif q_position in {24}:
            return k_position == 11
        elif q_position in {25}:
            return k_position == 27
        elif q_position in {27}:
            return k_position == 20
        elif q_position in {28}:
            return k_position == 36
        elif q_position in {32}:
            return k_position == 15
        elif q_position in {33}:
            return k_position == 1
        elif q_position in {38}:
            return k_position == 24
        elif q_position in {41}:
            return k_position == 40
        elif q_position in {43}:
            return k_position == 14
        elif q_position in {48}:
            return k_position == 30

    attn_1_3_pattern = select_closest(positions, positions, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_0_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(mlp_0_0_output, attn_0_2_output):
        if mlp_0_0_output in {
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
            23,
            24,
            25,
            26,
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
            return attn_0_2_output == ""
        elif mlp_0_0_output in {27, 11, 22}:
            return attn_0_2_output == "<pad>"

    num_attn_1_0_pattern = select(attn_0_2_outputs, mlp_0_0_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_3_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(q_token, k_token):
        if q_token in {"3", "0", "1", "5", "2", "4", "<s>"}:
            return k_token == ""

    num_attn_1_1_pattern = select(tokens, tokens, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_3_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(token, attn_0_1_output):
        if token in {"0", "1", "5", "2", "4", "<s>"}:
            return attn_0_1_output == ""
        elif token in {"3"}:
            return attn_0_1_output == "<pad>"

    num_attn_1_2_pattern = select(attn_0_1_outputs, tokens, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, num_attn_0_3_outputs)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(attn_0_1_output, token):
        if attn_0_1_output in {"0"}:
            return token == "0"
        elif attn_0_1_output in {"1"}:
            return token == "1"
        elif attn_0_1_output in {"2"}:
            return token == "2"
        elif attn_0_1_output in {"5", "<s>", "3"}:
            return token == ""
        elif attn_0_1_output in {"4"}:
            return token == "4"

    num_attn_1_3_pattern = select(tokens, attn_0_1_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, ones)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_0_0_output, attn_1_3_output):
        key = (attn_0_0_output, attn_1_3_output)
        if key in {
            ("0", "0"),
            ("0", "1"),
            ("0", "3"),
            ("0", "4"),
            ("0", "5"),
            ("1", "0"),
            ("1", "1"),
            ("1", "3"),
            ("1", "4"),
            ("1", "5"),
            ("2", "5"),
            ("3", "0"),
            ("3", "1"),
            ("3", "3"),
            ("3", "5"),
            ("4", "5"),
            ("5", "0"),
            ("5", "1"),
            ("5", "2"),
            ("5", "3"),
            ("5", "4"),
            ("5", "5"),
            ("<s>", "5"),
        }:
            return 4
        return 7

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_1_3_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_0_2_output, num_mlp_0_0_output):
        key = (attn_0_2_output, num_mlp_0_0_output)
        if key in {
            ("0", 48),
            ("1", 39),
            ("1", 48),
            ("3", 48),
            ("5", 39),
            ("5", 48),
            ("<s>", 39),
            ("<s>", 48),
        }:
            return 11
        return 32

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_0_2_outputs, num_mlp_0_0_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_0_output, num_attn_0_3_output):
        key = (num_attn_1_0_output, num_attn_0_3_output)
        return 29

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_0_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_1_output, num_attn_1_0_output):
        key = (num_attn_0_1_output, num_attn_1_0_output)
        if key in {
            (0, 53),
            (0, 54),
            (0, 55),
            (0, 56),
            (0, 57),
            (0, 58),
            (0, 59),
            (0, 60),
            (0, 61),
            (0, 62),
            (0, 63),
            (0, 64),
            (0, 65),
            (0, 66),
            (0, 67),
            (0, 68),
            (0, 69),
            (0, 70),
            (0, 71),
            (0, 72),
            (0, 73),
            (0, 74),
            (0, 75),
            (0, 76),
            (1, 66),
            (1, 67),
            (1, 68),
            (1, 69),
            (1, 70),
            (1, 71),
            (1, 72),
            (1, 73),
            (1, 74),
            (1, 75),
            (1, 76),
            (1, 77),
            (1, 78),
            (1, 79),
            (1, 80),
            (1, 81),
            (1, 82),
            (1, 83),
            (1, 84),
            (1, 85),
            (1, 86),
            (1, 87),
            (1, 88),
            (1, 89),
            (2, 79),
            (2, 80),
            (2, 81),
            (2, 82),
            (2, 83),
            (2, 84),
            (2, 85),
            (2, 86),
            (2, 87),
            (2, 88),
            (2, 89),
            (2, 90),
            (2, 91),
            (2, 92),
            (2, 93),
            (2, 94),
            (2, 95),
            (2, 96),
            (2, 97),
            (2, 98),
            (2, 99),
            (3, 92),
            (3, 93),
            (3, 94),
            (3, 95),
            (3, 96),
            (3, 97),
            (3, 98),
            (3, 99),
        }:
            return 24
        elif key in {
            (0, 77),
            (0, 78),
            (0, 79),
            (0, 80),
            (0, 81),
            (0, 82),
            (0, 83),
            (0, 84),
            (0, 85),
            (0, 86),
            (0, 87),
            (0, 88),
            (0, 89),
            (0, 90),
            (0, 91),
            (0, 92),
            (0, 93),
            (0, 94),
            (0, 95),
            (0, 96),
            (0, 97),
            (0, 98),
            (0, 99),
            (1, 90),
            (1, 91),
            (1, 92),
            (1, 93),
            (1, 94),
            (1, 95),
            (1, 96),
            (1, 97),
            (1, 98),
            (1, 99),
        }:
            return 35
        return 13

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_token, k_token):
        if q_token in {"2", "0", "5", "3"}:
            return k_token == "1"
        elif q_token in {"4", "1"}:
            return k_token == "0"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_0_pattern = select_closest(tokens, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_2_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_token, k_token):
        if q_token in {"0", "4"}:
            return k_token == "1"
        elif q_token in {"2", "1", "<s>"}:
            return k_token == "3"
        elif q_token in {"3"}:
            return k_token == "2"
        elif q_token in {"5"}:
            return k_token == "5"

    attn_2_1_pattern = select_closest(tokens, tokens, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, mlp_1_0_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_token, k_token):
        if q_token in {"2", "0", "5"}:
            return k_token == "4"
        elif q_token in {"4", "1", "3"}:
            return k_token == "5"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_2_pattern = select_closest(tokens, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_1_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_token, k_token):
        if q_token in {"0", "1", "4", "3"}:
            return k_token == "5"
        elif q_token in {"2", "5"}:
            return k_token == "0"
        elif q_token in {"<s>"}:
            return k_token == "3"

    attn_2_3_pattern = select_closest(tokens, tokens, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_0_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_0_0_output, attn_1_3_output):
        if attn_0_0_output in {"3", "0", "1", "5", "2", "4", "<s>"}:
            return attn_1_3_output == ""

    num_attn_2_0_pattern = select(attn_1_3_outputs, attn_0_0_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_2_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(token, attn_1_2_output):
        if token in {"0"}:
            return attn_1_2_output == 22
        elif token in {"1"}:
            return attn_1_2_output == 19
        elif token in {"2"}:
            return attn_1_2_output == 9
        elif token in {"3"}:
            return attn_1_2_output == 39
        elif token in {"4", "<s>"}:
            return attn_1_2_output == 41
        elif token in {"5"}:
            return attn_1_2_output == 46

    num_attn_2_1_pattern = select(attn_1_2_outputs, tokens, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, ones)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_2_output, attn_0_0_output):
        if attn_1_2_output in {
            0,
            1,
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
            return attn_0_0_output == ""
        elif attn_1_2_output in {16, 2}:
            return attn_0_0_output == "<pad>"

    num_attn_2_2_pattern = select(attn_0_0_outputs, attn_1_2_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_1_2_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_1_2_output, attn_0_1_output):
        if attn_1_2_output in {
            0,
            1,
            3,
            5,
            6,
            8,
            9,
            10,
            11,
            12,
            13,
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
            return attn_0_1_output == ""
        elif attn_1_2_output in {2, 4, 7, 15, 26}:
            return attn_0_1_output == "<pad>"

    num_attn_2_3_pattern = select(attn_0_1_outputs, attn_1_2_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_0_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_2_1_output, attn_0_2_output):
        key = (attn_2_1_output, attn_0_2_output)
        if key in {
            (3, "2"),
            (3, "<s>"),
            (5, "<s>"),
            (6, "<s>"),
            (7, "2"),
            (7, "<s>"),
            (8, "<s>"),
            (9, "2"),
            (9, "<s>"),
            (10, "<s>"),
            (11, "<s>"),
            (13, "<s>"),
            (17, "<s>"),
            (18, "2"),
            (18, "<s>"),
            (19, "2"),
            (19, "<s>"),
            (22, "2"),
            (22, "<s>"),
            (23, "2"),
            (23, "<s>"),
            (25, "2"),
            (25, "<s>"),
            (26, "<s>"),
            (27, "<s>"),
            (30, "2"),
            (30, "<s>"),
            (33, "2"),
            (33, "<s>"),
            (34, "<s>"),
            (35, "<s>"),
            (38, "2"),
            (38, "<s>"),
            (39, "2"),
            (39, "<s>"),
            (40, "<s>"),
            (41, "2"),
            (41, "<s>"),
            (43, "2"),
            (43, "<s>"),
            (45, "<s>"),
            (47, "<s>"),
            (48, "<s>"),
        }:
            return 29
        elif key in {(33, "0"), (33, "3")}:
            return 15
        return 10

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_2_1_outputs, attn_0_2_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_0_0_output, attn_1_3_output):
        key = (attn_0_0_output, attn_1_3_output)
        if key in {("2", "2"), ("<s>", "3")}:
            return 4
        elif key in {
            ("1", "1"),
            ("4", "1"),
            ("<s>", "0"),
            ("<s>", "1"),
            ("<s>", "2"),
            ("<s>", "4"),
            ("<s>", "5"),
            ("<s>", "<s>"),
        }:
            return 14
        return 35

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_0_0_outputs, attn_1_3_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_1_3_output, num_attn_2_3_output):
        key = (num_attn_1_3_output, num_attn_2_3_output)
        return 3

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_2_3_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_1_1_output, num_attn_1_3_output):
        key = (num_attn_1_1_output, num_attn_1_3_output)
        return 18

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_1_1_outputs, num_attn_1_3_outputs)
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
            "3",
            "4",
            "0",
            "1",
            "3",
            "5",
            "0",
            "0",
            "1",
            "4",
            "5",
            "4",
            "1",
            "2",
            "4",
            "5",
            "2",
            "4",
            "3",
            "4",
            "2",
            "4",
            "5",
            "2",
            "4",
            "1",
            "1",
            "0",
            "5",
            "1",
            "1",
            "5",
            "1",
            "1",
            "0",
            "4",
            "1",
            "0",
        ]
    )
)
