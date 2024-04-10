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
        "output/length/rasp/double_hist/trainlength30/s4/double_hist_weights.csv",
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
            return k_token == "3"
        elif q_token in {"<s>", "1"}:
            return k_token == "0"
        elif q_token in {"2"}:
            return k_token == "4"
        elif q_token in {"4", "3"}:
            return k_token == "2"
        elif q_token in {"5"}:
            return k_token == "1"

    attn_0_0_pattern = select_closest(tokens, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, positions)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 3
        elif q_position in {1, 2, 36}:
            return k_position == 6
        elif q_position in {3, 4, 5, 39, 31}:
            return k_position == 5
        elif q_position in {10, 6, 7}:
            return k_position == 7
        elif q_position in {8, 12}:
            return k_position == 1
        elif q_position in {9, 35, 30}:
            return k_position == 9
        elif q_position in {11}:
            return k_position == 2
        elif q_position in {32, 38, 13, 14, 16, 17, 23}:
            return k_position == 8
        elif q_position in {15}:
            return k_position == 10
        elif q_position in {18, 34, 37}:
            return k_position == 12
        elif q_position in {19, 21, 22}:
            return k_position == 11
        elif q_position in {24, 20}:
            return k_position == 15
        elif q_position in {25, 26}:
            return k_position == 16
        elif q_position in {27, 28, 29}:
            return k_position == 17
        elif q_position in {33}:
            return k_position == 4

    attn_0_1_pattern = select_closest(positions, positions, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, positions)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "2"
        elif q_token in {"5", "4", "1", "2"}:
            return k_token == "1"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"<s>"}:
            return k_token == "0"

    attn_0_2_pattern = select_closest(tokens, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, tokens)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_token, k_token):
        if q_token in {"0", "<s>"}:
            return k_token == "3"
        elif q_token in {"5", "1", "2"}:
            return k_token == "1"
        elif q_token in {"3"}:
            return k_token == "4"
        elif q_token in {"4"}:
            return k_token == "2"

    attn_0_3_pattern = select_closest(tokens, tokens, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, tokens)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_token, k_token):
        if q_token in {"0", "2", "4", "5", "3", "1", "<s>"}:
            return k_token == ""

    num_attn_0_0_pattern = select(tokens, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_position, k_position):
        if q_position in {0}:
            return k_position == 20
        elif q_position in {1, 27}:
            return k_position == 28
        elif q_position in {8, 2}:
            return k_position == 17
        elif q_position in {18, 3, 5}:
            return k_position == 24
        elif q_position in {32, 4}:
            return k_position == 21
        elif q_position in {6, 15}:
            return k_position == 25
        elif q_position in {7}:
            return k_position == 26
        elif q_position in {9}:
            return k_position == 16
        elif q_position in {10}:
            return k_position == 23
        elif q_position in {11, 13}:
            return k_position == 18
        elif q_position in {12, 39}:
            return k_position == 15
        elif q_position in {20, 14, 30}:
            return k_position == 22
        elif q_position in {16, 29, 23}:
            return k_position == 34
        elif q_position in {17, 28, 21}:
            return k_position == 33
        elif q_position in {19}:
            return k_position == 31
        elif q_position in {25, 22}:
            return k_position == 30
        elif q_position in {24, 26}:
            return k_position == 27
        elif q_position in {31}:
            return k_position == 0
        elif q_position in {33, 34, 35}:
            return k_position == 11
        elif q_position in {36}:
            return k_position == 8
        elif q_position in {37}:
            return k_position == 39
        elif q_position in {38}:
            return k_position == 10

    num_attn_0_1_pattern = select(positions, positions, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_position, k_position):
        if q_position in {0, 5, 6, 7, 9}:
            return k_position == 19
        elif q_position in {1}:
            return k_position == 17
        elif q_position in {2}:
            return k_position == 21
        elif q_position in {3}:
            return k_position == 15
        elif q_position in {33, 4, 12}:
            return k_position == 26
        elif q_position in {8, 21}:
            return k_position == 24
        elif q_position in {26, 10, 19}:
            return k_position == 37
        elif q_position in {11, 23}:
            return k_position == 30
        elif q_position in {37, 13, 22}:
            return k_position == 29
        elif q_position in {20, 14}:
            return k_position == 27
        elif q_position in {15}:
            return k_position == 34
        elif q_position in {16}:
            return k_position == 32
        elif q_position in {17, 30}:
            return k_position == 22
        elif q_position in {25, 18}:
            return k_position == 39
        elif q_position in {24}:
            return k_position == 36
        elif q_position in {35, 27}:
            return k_position == 35
        elif q_position in {28}:
            return k_position == 23
        elif q_position in {29}:
            return k_position == 0
        elif q_position in {38, 31}:
            return k_position == 9
        elif q_position in {32}:
            return k_position == 18
        elif q_position in {34}:
            return k_position == 8
        elif q_position in {36}:
            return k_position == 33
        elif q_position in {39}:
            return k_position == 28

    num_attn_0_2_pattern = select(positions, positions, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_token, k_token):
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

    num_attn_0_3_pattern = select(tokens, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(token):
        key = token
        if key in {"", "4"}:
            return 26
        elif key in {""}:
            return 8
        return 2

    mlp_0_0_outputs = [mlp_0_0(k0) for k0 in tokens]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_2_output, attn_0_3_output):
        key = (attn_0_2_output, attn_0_3_output)
        if key in {
            ("0", "1"),
            ("0", "4"),
            ("0", "5"),
            ("1", "0"),
            ("1", "1"),
            ("1", "2"),
            ("1", "4"),
            ("1", "5"),
            ("2", "4"),
            ("3", "4"),
            ("4", "1"),
            ("4", "2"),
            ("4", "4"),
            ("4", "5"),
            ("5", "0"),
            ("5", "1"),
            ("5", "2"),
            ("5", "4"),
            ("5", "5"),
            ("<s>", "4"),
        }:
            return 22
        return 26

    mlp_0_1_outputs = [
        mlp_0_1(k0, k1) for k0, k1 in zip(attn_0_2_outputs, attn_0_3_outputs)
    ]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_0_output):
        key = num_attn_0_0_output
        return 21

    num_mlp_0_0_outputs = [num_mlp_0_0(k0) for k0 in num_attn_0_0_outputs]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_0_output):
        key = num_attn_0_0_output
        return 21

    num_mlp_0_1_outputs = [num_mlp_0_1(k0) for k0 in num_attn_0_0_outputs]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(attn_0_2_output, token):
        if attn_0_2_output in {"0", "2", "5", "4", "<s>"}:
            return token == ""
        elif attn_0_2_output in {"1"}:
            return token == "3"
        elif attn_0_2_output in {"3"}:
            return token == "2"

    attn_1_0_pattern = select_closest(tokens, attn_0_2_outputs, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, attn_0_2_outputs)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(attn_0_1_output, position):
        if attn_0_1_output in {0, 3, 38}:
            return position == 4
        elif attn_0_1_output in {1}:
            return position == 37
        elif attn_0_1_output in {2}:
            return position == 10
        elif attn_0_1_output in {32, 33, 34, 4, 36, 39, 30, 31}:
            return position == 5
        elif attn_0_1_output in {5, 6}:
            return position == 6
        elif attn_0_1_output in {10, 7}:
            return position == 7
        elif attn_0_1_output in {8, 12}:
            return position == 9
        elif attn_0_1_output in {9}:
            return position == 1
        elif attn_0_1_output in {11}:
            return position == 12
        elif attn_0_1_output in {13}:
            return position == 14
        elif attn_0_1_output in {17, 14}:
            return position == 16
        elif attn_0_1_output in {19, 15}:
            return position == 25
        elif attn_0_1_output in {16, 23}:
            return position == 15
        elif attn_0_1_output in {18}:
            return position == 20
        elif attn_0_1_output in {24, 26, 20}:
            return position == 21
        elif attn_0_1_output in {29, 21}:
            return position == 22
        elif attn_0_1_output in {27, 22}:
            return position == 28
        elif attn_0_1_output in {25, 28}:
            return position == 19
        elif attn_0_1_output in {35}:
            return position == 35
        elif attn_0_1_output in {37}:
            return position == 24

    attn_1_1_pattern = select_closest(positions, attn_0_1_outputs, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, attn_0_1_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(token, mlp_0_1_output):
        if token in {"0"}:
            return mlp_0_1_output == 1
        elif token in {"5", "1"}:
            return mlp_0_1_output == 22
        elif token in {"2"}:
            return mlp_0_1_output == 3
        elif token in {"3"}:
            return mlp_0_1_output == 4
        elif token in {"4"}:
            return mlp_0_1_output == 24
        elif token in {"<s>"}:
            return mlp_0_1_output == 26

    attn_1_2_pattern = select_closest(mlp_0_1_outputs, tokens, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, mlp_0_1_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(attn_0_1_output, position):
        if attn_0_1_output in {0, 34, 3, 35, 36, 37, 30, 31}:
            return position == 4
        elif attn_0_1_output in {1, 4}:
            return position == 5
        elif attn_0_1_output in {2, 10}:
            return position == 11
        elif attn_0_1_output in {5}:
            return position == 6
        elif attn_0_1_output in {9, 6}:
            return position == 1
        elif attn_0_1_output in {8, 13, 7}:
            return position == 8
        elif attn_0_1_output in {11}:
            return position == 10
        elif attn_0_1_output in {19, 12, 22, 23}:
            return position == 13
        elif attn_0_1_output in {18, 14}:
            return position == 15
        elif attn_0_1_output in {15}:
            return position == 19
        elif attn_0_1_output in {16, 21}:
            return position == 21
        elif attn_0_1_output in {17}:
            return position == 7
        elif attn_0_1_output in {20, 29}:
            return position == 20
        elif attn_0_1_output in {24}:
            return position == 25
        elif attn_0_1_output in {25, 27}:
            return position == 22
        elif attn_0_1_output in {26, 28}:
            return position == 17
        elif attn_0_1_output in {32}:
            return position == 28
        elif attn_0_1_output in {33}:
            return position == 9
        elif attn_0_1_output in {38}:
            return position == 23
        elif attn_0_1_output in {39}:
            return position == 34

    attn_1_3_pattern = select_closest(positions, attn_0_1_outputs, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_1_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(mlp_0_1_output, token):
        if mlp_0_1_output in {0}:
            return token == "<pad>"
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

    num_attn_1_0_pattern = select(tokens, mlp_0_1_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, num_attn_0_0_outputs)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_3_output, token):
        if attn_0_3_output in {"0"}:
            return token == "0"
        elif attn_0_3_output in {"1"}:
            return token == "1"
        elif attn_0_3_output in {"2"}:
            return token == "2"
        elif attn_0_3_output in {"3"}:
            return token == "3"
        elif attn_0_3_output in {"4"}:
            return token == "4"
        elif attn_0_3_output in {"5", "<s>"}:
            return token == "5"

    num_attn_1_1_pattern = select(tokens, attn_0_3_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, ones)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_3_output, token):
        if attn_0_3_output in {"0", "2", "4", "5", "3", "1", "<s>"}:
            return token == ""

    num_attn_1_2_pattern = select(tokens, attn_0_3_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, ones)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(q_mlp_0_1_output, k_mlp_0_1_output):
        if q_mlp_0_1_output in {0, 1, 2, 33}:
            return k_mlp_0_1_output == 31
        elif q_mlp_0_1_output in {18, 3, 12, 21}:
            return k_mlp_0_1_output == 19
        elif q_mlp_0_1_output in {32, 17, 4}:
            return k_mlp_0_1_output == 35
        elif q_mlp_0_1_output in {8, 5}:
            return k_mlp_0_1_output == 23
        elif q_mlp_0_1_output in {6}:
            return k_mlp_0_1_output == 9
        elif q_mlp_0_1_output in {35, 38, 7}:
            return k_mlp_0_1_output == 22
        elif q_mlp_0_1_output in {9, 19, 29}:
            return k_mlp_0_1_output == 33
        elif q_mlp_0_1_output in {10}:
            return k_mlp_0_1_output == 7
        elif q_mlp_0_1_output in {11}:
            return k_mlp_0_1_output == 28
        elif q_mlp_0_1_output in {36, 13}:
            return k_mlp_0_1_output == 36
        elif q_mlp_0_1_output in {14}:
            return k_mlp_0_1_output == 25
        elif q_mlp_0_1_output in {37, 15, 24, 25, 27, 30}:
            return k_mlp_0_1_output == 30
        elif q_mlp_0_1_output in {16, 39}:
            return k_mlp_0_1_output == 38
        elif q_mlp_0_1_output in {20}:
            return k_mlp_0_1_output == 37
        elif q_mlp_0_1_output in {22}:
            return k_mlp_0_1_output == 39
        elif q_mlp_0_1_output in {23}:
            return k_mlp_0_1_output == 27
        elif q_mlp_0_1_output in {26}:
            return k_mlp_0_1_output == 2
        elif q_mlp_0_1_output in {28}:
            return k_mlp_0_1_output == 32
        elif q_mlp_0_1_output in {31}:
            return k_mlp_0_1_output == 29
        elif q_mlp_0_1_output in {34}:
            return k_mlp_0_1_output == 34

    num_attn_1_3_pattern = select(mlp_0_1_outputs, mlp_0_1_outputs, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_2_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(num_mlp_0_0_output, attn_0_3_output):
        key = (num_mlp_0_0_output, attn_0_3_output)
        return 25

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(num_mlp_0_0_outputs, attn_0_3_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(num_mlp_0_1_output, num_mlp_0_0_output):
        key = (num_mlp_0_1_output, num_mlp_0_0_output)
        return 5

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(num_mlp_0_1_outputs, num_mlp_0_0_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_3_output):
        key = num_attn_1_3_output
        return 4

    num_mlp_1_0_outputs = [num_mlp_1_0(k0) for k0 in num_attn_1_3_outputs]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_1_2_output, num_attn_0_1_output):
        key = (num_attn_1_2_output, num_attn_0_1_output)
        return 11

    num_mlp_1_1_outputs = [
        num_mlp_1_1(k0, k1)
        for k0, k1 in zip(num_attn_1_2_outputs, num_attn_0_1_outputs)
    ]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_token, k_token):
        if q_token in {"0", "1"}:
            return k_token == "2"
        elif q_token in {"3", "2"}:
            return k_token == "1"
        elif q_token in {"4"}:
            return k_token == "0"
        elif q_token in {"5"}:
            return k_token == "3"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_0_pattern = select_closest(tokens, tokens, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_0_0_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_token, k_token):
        if q_token in {"0", "2"}:
            return k_token == "5"
        elif q_token in {"1"}:
            return k_token == "2"
        elif q_token in {"5", "4", "3"}:
            return k_token == "0"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_1_pattern = select_closest(tokens, tokens, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_3_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_token, k_token):
        if q_token in {"0", "2", "4", "3", "1"}:
            return k_token == "5"
        elif q_token in {"5"}:
            return k_token == "4"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_2_pattern = select_closest(tokens, tokens, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_1_2_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(token, attn_0_2_output):
        if token in {"0", "4", "<s>"}:
            return attn_0_2_output == ""
        elif token in {"3", "5", "1"}:
            return attn_0_2_output == "4"
        elif token in {"2"}:
            return attn_0_2_output == "3"

    attn_2_3_pattern = select_closest(attn_0_2_outputs, tokens, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_3_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(attn_1_2_output, attn_0_3_output):
        if attn_1_2_output in {
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
            return attn_0_3_output == ""

    num_attn_2_0_pattern = select(attn_0_3_outputs, attn_1_2_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_1_0_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_1_1_output, attn_1_3_output):
        if attn_1_1_output in {0, 11, 17, 21, 27}:
            return attn_1_3_output == 32
        elif attn_1_1_output in {1, 14, 25}:
            return attn_1_3_output == 34
        elif attn_1_1_output in {32, 9, 2, 33}:
            return attn_1_3_output == 33
        elif attn_1_1_output in {3}:
            return attn_1_3_output == 3
        elif attn_1_1_output in {4}:
            return attn_1_3_output == 4
        elif attn_1_1_output in {5}:
            return attn_1_3_output == 29
        elif attn_1_1_output in {6, 12, 16, 26, 29, 31}:
            return attn_1_3_output == 36
        elif attn_1_1_output in {7}:
            return attn_1_3_output == 6
        elif attn_1_1_output in {8, 34, 20, 15}:
            return attn_1_3_output == 30
        elif attn_1_1_output in {10, 38}:
            return attn_1_3_output == 35
        elif attn_1_1_output in {37, 35, 13, 30}:
            return attn_1_3_output == 37
        elif attn_1_1_output in {24, 18}:
            return attn_1_3_output == 38
        elif attn_1_1_output in {19, 36, 22, 23}:
            return attn_1_3_output == 39
        elif attn_1_1_output in {28}:
            return attn_1_3_output == 27
        elif attn_1_1_output in {39}:
            return attn_1_3_output == 31

    num_attn_2_1_pattern = select(attn_1_3_outputs, attn_1_1_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_1_2_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(q_attn_0_2_output, k_attn_0_2_output):
        if q_attn_0_2_output in {"0", "2", "4", "5", "3", "1", "<s>"}:
            return k_attn_0_2_output == ""

    num_attn_2_2_pattern = select(attn_0_2_outputs, attn_0_2_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_0_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_0_2_output, token):
        if attn_0_2_output in {"0", "2", "5", "3", "1", "<s>"}:
            return token == ""
        elif attn_0_2_output in {"4"}:
            return token == "2"

    num_attn_2_3_pattern = select(tokens, attn_0_2_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_0_2_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(num_mlp_0_1_output, num_mlp_1_0_output):
        key = (num_mlp_0_1_output, num_mlp_1_0_output)
        if key in {
            (0, 34),
            (1, 34),
            (2, 34),
            (7, 34),
            (8, 34),
            (11, 34),
            (12, 34),
            (14, 34),
            (16, 10),
            (16, 18),
            (16, 24),
            (16, 31),
            (16, 34),
            (17, 34),
            (21, 34),
            (23, 34),
            (25, 34),
            (26, 34),
            (28, 34),
            (29, 34),
            (31, 34),
            (32, 34),
            (34, 10),
            (34, 18),
            (34, 24),
            (34, 31),
            (34, 34),
            (36, 34),
            (37, 34),
            (38, 10),
            (38, 18),
            (38, 31),
            (38, 34),
            (39, 34),
        }:
            return 32
        return 22

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(num_mlp_0_1_outputs, num_mlp_1_0_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(mlp_1_1_output, num_mlp_0_1_output):
        key = (mlp_1_1_output, num_mlp_0_1_output)
        if key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 6),
            (0, 7),
            (0, 9),
            (0, 10),
            (0, 12),
            (0, 16),
            (0, 17),
            (0, 18),
            (0, 20),
            (0, 23),
            (0, 26),
            (0, 27),
            (0, 28),
            (0, 30),
            (0, 31),
            (0, 32),
            (0, 33),
            (0, 34),
            (0, 35),
            (0, 37),
            (0, 38),
            (0, 39),
            (1, 10),
            (1, 32),
            (1, 37),
            (1, 38),
            (3, 32),
            (6, 10),
            (6, 32),
            (6, 37),
            (6, 38),
            (7, 10),
            (7, 32),
            (7, 37),
            (7, 38),
            (8, 32),
            (9, 32),
            (10, 32),
            (11, 32),
            (14, 0),
            (14, 1),
            (14, 2),
            (14, 3),
            (14, 4),
            (14, 6),
            (14, 7),
            (14, 9),
            (14, 10),
            (14, 12),
            (14, 16),
            (14, 17),
            (14, 18),
            (14, 20),
            (14, 23),
            (14, 26),
            (14, 27),
            (14, 28),
            (14, 30),
            (14, 31),
            (14, 32),
            (14, 33),
            (14, 34),
            (14, 35),
            (14, 37),
            (14, 38),
            (14, 39),
            (15, 0),
            (15, 1),
            (15, 2),
            (15, 3),
            (15, 4),
            (15, 6),
            (15, 7),
            (15, 9),
            (15, 10),
            (15, 12),
            (15, 16),
            (15, 17),
            (15, 18),
            (15, 20),
            (15, 23),
            (15, 26),
            (15, 27),
            (15, 28),
            (15, 30),
            (15, 31),
            (15, 32),
            (15, 33),
            (15, 34),
            (15, 35),
            (15, 37),
            (15, 38),
            (15, 39),
            (18, 10),
            (18, 32),
            (18, 37),
            (18, 38),
            (19, 0),
            (19, 1),
            (19, 2),
            (19, 7),
            (19, 10),
            (19, 12),
            (19, 16),
            (19, 17),
            (19, 18),
            (19, 23),
            (19, 32),
            (19, 33),
            (19, 35),
            (19, 37),
            (19, 38),
            (19, 39),
            (22, 10),
            (22, 32),
            (22, 37),
            (22, 38),
            (23, 32),
            (23, 37),
            (23, 38),
            (26, 0),
            (26, 1),
            (26, 2),
            (26, 3),
            (26, 4),
            (26, 5),
            (26, 6),
            (26, 7),
            (26, 9),
            (26, 10),
            (26, 12),
            (26, 16),
            (26, 17),
            (26, 18),
            (26, 19),
            (26, 20),
            (26, 23),
            (26, 26),
            (26, 27),
            (26, 28),
            (26, 29),
            (26, 30),
            (26, 31),
            (26, 32),
            (26, 33),
            (26, 34),
            (26, 35),
            (26, 37),
            (26, 38),
            (26, 39),
            (27, 32),
            (32, 10),
            (32, 17),
            (32, 18),
            (32, 32),
            (32, 35),
            (32, 37),
            (32, 38),
            (36, 32),
            (39, 32),
            (39, 38),
        }:
            return 23
        return 1

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(mlp_1_1_outputs, num_mlp_0_1_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_0_1_output, num_attn_1_3_output):
        key = (num_attn_0_1_output, num_attn_1_3_output)
        return 23

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_1_3_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_3_output, num_attn_2_1_output):
        key = (num_attn_2_3_output, num_attn_2_1_output)
        return 13

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
            "4",
            "3",
            "0",
            "5",
            "5",
            "1",
            "5",
            "0",
            "2",
            "5",
            "0",
            "1",
        ]
    )
)
