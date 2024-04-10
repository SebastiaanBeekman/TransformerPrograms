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
        "output/length/rasp/double_hist/trainlength40/s2/double_hist_weights.csv",
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
        if token in {"1", "2", "0", "5"}:
            return position == 8
        elif token in {"3", "4"}:
            return position == 9
        elif token in {"<s>"}:
            return position == 5

    attn_0_0_pattern = select_closest(positions, tokens, predicate_0_0)
    attn_0_0_outputs = aggregate(attn_0_0_pattern, positions)
    attn_0_0_output_scores = classifier_weights.loc[
        [("attn_0_0_outputs", str(v)) for v in attn_0_0_outputs]
    ]

    # attn_0_1 ####################################################
    def predicate_0_1(q_token, k_token):
        if q_token in {"2", "3", "0", "4"}:
            return k_token == "4"
        elif q_token in {"1"}:
            return k_token == "5"
        elif q_token in {"5"}:
            return k_token == "0"
        elif q_token in {"<s>"}:
            return k_token == "3"

    attn_0_1_pattern = select_closest(tokens, tokens, predicate_0_1)
    attn_0_1_outputs = aggregate(attn_0_1_pattern, tokens)
    attn_0_1_output_scores = classifier_weights.loc[
        [("attn_0_1_outputs", str(v)) for v in attn_0_1_outputs]
    ]

    # attn_0_2 ####################################################
    def predicate_0_2(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "2"
        elif q_token in {"1"}:
            return k_token == "4"
        elif q_token in {"2", "5"}:
            return k_token == "1"
        elif q_token in {"3", "4"}:
            return k_token == "0"
        elif q_token in {"<s>"}:
            return k_token == "5"

    attn_0_2_pattern = select_closest(tokens, tokens, predicate_0_2)
    attn_0_2_outputs = aggregate(attn_0_2_pattern, positions)
    attn_0_2_output_scores = classifier_weights.loc[
        [("attn_0_2_outputs", str(v)) for v in attn_0_2_outputs]
    ]

    # attn_0_3 ####################################################
    def predicate_0_3(q_position, k_position):
        if q_position in {0, 1, 4}:
            return k_position == 5
        elif q_position in {9, 2, 3, 5}:
            return k_position == 6
        elif q_position in {10, 6}:
            return k_position == 1
        elif q_position in {7, 11, 12, 14, 20, 21}:
            return k_position == 30
        elif q_position in {8, 18, 15}:
            return k_position == 31
        elif q_position in {13, 38}:
            return k_position == 18
        elif q_position in {16, 19, 28}:
            return k_position == 35
        elif q_position in {17}:
            return k_position == 25
        elif q_position in {22}:
            return k_position == 36
        elif q_position in {23}:
            return k_position == 37
        elif q_position in {24}:
            return k_position == 14
        elif q_position in {25}:
            return k_position == 19
        elif q_position in {26}:
            return k_position == 13
        elif q_position in {34, 48, 27, 30, 31}:
            return k_position == 21
        elif q_position in {43, 29}:
            return k_position == 24
        elif q_position in {32, 33, 35, 37}:
            return k_position == 17
        elif q_position in {36}:
            return k_position == 28
        elif q_position in {45, 46, 39}:
            return k_position == 23
        elif q_position in {40}:
            return k_position == 45
        elif q_position in {41, 44}:
            return k_position == 10
        elif q_position in {42}:
            return k_position == 48
        elif q_position in {47}:
            return k_position == 12
        elif q_position in {49}:
            return k_position == 15

    attn_0_3_pattern = select_closest(positions, positions, predicate_0_3)
    attn_0_3_outputs = aggregate(attn_0_3_pattern, positions)
    attn_0_3_output_scores = classifier_weights.loc[
        [("attn_0_3_outputs", str(v)) for v in attn_0_3_outputs]
    ]

    # num_attn_0_0 ####################################################
    def num_predicate_0_0(q_token, k_token):
        if q_token in {"3", "1", "0", "5", "4", "2", "<s>"}:
            return k_token == ""

    num_attn_0_0_pattern = select(tokens, tokens, num_predicate_0_0)
    num_attn_0_0_outputs = aggregate_sum(num_attn_0_0_pattern, ones)
    num_attn_0_0_output_scores = classifier_weights.loc[
        [("num_attn_0_0_outputs", "_") for v in num_attn_0_0_outputs]
    ].mul(num_attn_0_0_outputs, axis=0)

    # num_attn_0_1 ####################################################
    def num_predicate_0_1(q_token, k_token):
        if q_token in {"3", "0"}:
            return k_token == "<pad>"
        elif q_token in {"2", "1", "5"}:
            return k_token == ""
        elif q_token in {"<s>", "4"}:
            return k_token == "<s>"

    num_attn_0_1_pattern = select(tokens, tokens, num_predicate_0_1)
    num_attn_0_1_outputs = aggregate_sum(num_attn_0_1_pattern, ones)
    num_attn_0_1_output_scores = classifier_weights.loc[
        [("num_attn_0_1_outputs", "_") for v in num_attn_0_1_outputs]
    ].mul(num_attn_0_1_outputs, axis=0)

    # num_attn_0_2 ####################################################
    def num_predicate_0_2(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "0"
        elif q_token in {"1", "<s>"}:
            return k_token == "1"
        elif q_token in {"2"}:
            return k_token == "2"
        elif q_token in {"3"}:
            return k_token == "3"
        elif q_token in {"4"}:
            return k_token == "4"
        elif q_token in {"5"}:
            return k_token == "5"

    num_attn_0_2_pattern = select(tokens, tokens, num_predicate_0_2)
    num_attn_0_2_outputs = aggregate_sum(num_attn_0_2_pattern, ones)
    num_attn_0_2_output_scores = classifier_weights.loc[
        [("num_attn_0_2_outputs", "_") for v in num_attn_0_2_outputs]
    ].mul(num_attn_0_2_outputs, axis=0)

    # num_attn_0_3 ####################################################
    def num_predicate_0_3(q_token, k_token):
        if q_token in {"3", "1", "0", "5", "4", "2", "<s>"}:
            return k_token == ""

    num_attn_0_3_pattern = select(tokens, tokens, num_predicate_0_3)
    num_attn_0_3_outputs = aggregate_sum(num_attn_0_3_pattern, ones)
    num_attn_0_3_output_scores = classifier_weights.loc[
        [("num_attn_0_3_outputs", "_") for v in num_attn_0_3_outputs]
    ].mul(num_attn_0_3_outputs, axis=0)

    # mlp_0_0 #####################################################
    def mlp_0_0(token, attn_0_1_output):
        key = (token, attn_0_1_output)
        if key in {
            ("0", "0"),
            ("4", "4"),
            ("4", "<s>"),
            ("<s>", "0"),
            ("<s>", "1"),
            ("<s>", "2"),
            ("<s>", "3"),
            ("<s>", "4"),
            ("<s>", "5"),
            ("<s>", "<s>"),
        }:
            return 49
        return 1

    mlp_0_0_outputs = [mlp_0_0(k0, k1) for k0, k1 in zip(tokens, attn_0_1_outputs)]
    mlp_0_0_output_scores = classifier_weights.loc[
        [("mlp_0_0_outputs", str(v)) for v in mlp_0_0_outputs]
    ]

    # mlp_0_1 #####################################################
    def mlp_0_1(attn_0_1_output):
        key = attn_0_1_output
        if key in {""}:
            return 21
        elif key in {"<pad>"}:
            return 0
        elif key in {""}:
            return 27
        return 1

    mlp_0_1_outputs = [mlp_0_1(k0) for k0 in attn_0_1_outputs]
    mlp_0_1_output_scores = classifier_weights.loc[
        [("mlp_0_1_outputs", str(v)) for v in mlp_0_1_outputs]
    ]

    # num_mlp_0_0 #################################################
    def num_mlp_0_0(num_attn_0_1_output, num_attn_0_0_output):
        key = (num_attn_0_1_output, num_attn_0_0_output)
        return 26

    num_mlp_0_0_outputs = [
        num_mlp_0_0(k0, k1)
        for k0, k1 in zip(num_attn_0_1_outputs, num_attn_0_0_outputs)
    ]
    num_mlp_0_0_output_scores = classifier_weights.loc[
        [("num_mlp_0_0_outputs", str(v)) for v in num_mlp_0_0_outputs]
    ]

    # num_mlp_0_1 #################################################
    def num_mlp_0_1(num_attn_0_0_output, num_attn_0_3_output):
        key = (num_attn_0_0_output, num_attn_0_3_output)
        return 21

    num_mlp_0_1_outputs = [
        num_mlp_0_1(k0, k1)
        for k0, k1 in zip(num_attn_0_0_outputs, num_attn_0_3_outputs)
    ]
    num_mlp_0_1_output_scores = classifier_weights.loc[
        [("num_mlp_0_1_outputs", str(v)) for v in num_mlp_0_1_outputs]
    ]

    # attn_1_0 ####################################################
    def predicate_1_0(q_position, k_position):
        if q_position in {0}:
            return k_position == 3
        elif q_position in {1, 2, 3, 4, 5, 41, 42, 43, 44, 46, 48}:
            return k_position == 5
        elif q_position in {6}:
            return k_position == 7
        elif q_position in {7}:
            return k_position == 1
        elif q_position in {8, 9, 10, 11, 13, 14, 15, 17, 19}:
            return k_position == 2
        elif q_position in {33, 12}:
            return k_position == 13
        elif q_position in {16}:
            return k_position == 34
        elif q_position in {24, 49, 18, 47}:
            return k_position == 6
        elif q_position in {20, 23}:
            return k_position == 15
        elif q_position in {21, 22, 31}:
            return k_position == 10
        elif q_position in {32, 25, 26, 28}:
            return k_position == 14
        elif q_position in {27, 30}:
            return k_position == 17
        elif q_position in {29, 38}:
            return k_position == 18
        elif q_position in {34}:
            return k_position == 20
        elif q_position in {35}:
            return k_position == 19
        elif q_position in {36}:
            return k_position == 21
        elif q_position in {37}:
            return k_position == 23
        elif q_position in {39}:
            return k_position == 28
        elif q_position in {40, 45}:
            return k_position == 4

    attn_1_0_pattern = select_closest(positions, positions, predicate_1_0)
    attn_1_0_outputs = aggregate(attn_1_0_pattern, positions)
    attn_1_0_output_scores = classifier_weights.loc[
        [("attn_1_0_outputs", str(v)) for v in attn_1_0_outputs]
    ]

    # attn_1_1 ####################################################
    def predicate_1_1(q_token, k_token):
        if q_token in {"0"}:
            return k_token == "1"
        elif q_token in {"3", "1", "4"}:
            return k_token == "2"
        elif q_token in {"2", "5"}:
            return k_token == "3"
        elif q_token in {"<s>"}:
            return k_token == "5"

    attn_1_1_pattern = select_closest(tokens, tokens, predicate_1_1)
    attn_1_1_outputs = aggregate(attn_1_1_pattern, mlp_0_0_outputs)
    attn_1_1_output_scores = classifier_weights.loc[
        [("attn_1_1_outputs", str(v)) for v in attn_1_1_outputs]
    ]

    # attn_1_2 ####################################################
    def predicate_1_2(position, attn_0_2_output):
        if position in {0}:
            return attn_0_2_output == 35
        elif position in {1, 2, 31}:
            return attn_0_2_output == 11
        elif position in {16, 3, 22}:
            return attn_0_2_output == 14
        elif position in {4}:
            return attn_0_2_output == 29
        elif position in {5}:
            return attn_0_2_output == 18
        elif position in {10, 6}:
            return attn_0_2_output == 12
        elif position in {11, 7}:
            return attn_0_2_output == 0
        elif position in {8, 35}:
            return attn_0_2_output == 26
        elif position in {24, 9, 32, 27}:
            return attn_0_2_output == 25
        elif position in {12}:
            return attn_0_2_output == 23
        elif position in {13}:
            return attn_0_2_output == 24
        elif position in {33, 14}:
            return attn_0_2_output == 21
        elif position in {15}:
            return attn_0_2_output == 31
        elif position in {17, 45, 25}:
            return attn_0_2_output == 10
        elif position in {18}:
            return attn_0_2_output == 22
        elif position in {19, 21}:
            return attn_0_2_output == 6
        elif position in {20, 36}:
            return attn_0_2_output == 33
        elif position in {23}:
            return attn_0_2_output == 8
        elif position in {26}:
            return attn_0_2_output == 15
        elif position in {28}:
            return attn_0_2_output == 32
        elif position in {29}:
            return attn_0_2_output == 17
        elif position in {34, 30, 39}:
            return attn_0_2_output == 16
        elif position in {37}:
            return attn_0_2_output == 30
        elif position in {38}:
            return attn_0_2_output == 19
        elif position in {40, 48, 44, 46}:
            return attn_0_2_output == 1
        elif position in {41, 43, 49, 47}:
            return attn_0_2_output == 5
        elif position in {42}:
            return attn_0_2_output == 9

    attn_1_2_pattern = select_closest(attn_0_2_outputs, positions, predicate_1_2)
    attn_1_2_outputs = aggregate(attn_1_2_pattern, attn_0_2_outputs)
    attn_1_2_output_scores = classifier_weights.loc[
        [("attn_1_2_outputs", str(v)) for v in attn_1_2_outputs]
    ]

    # attn_1_3 ####################################################
    def predicate_1_3(token, position):
        if token in {"0"}:
            return position == 25
        elif token in {"1"}:
            return position == 26
        elif token in {"2", "5"}:
            return position == 30
        elif token in {"3"}:
            return position == 36
        elif token in {"<s>", "4"}:
            return position == 12

    attn_1_3_pattern = select_closest(positions, tokens, predicate_1_3)
    attn_1_3_outputs = aggregate(attn_1_3_pattern, attn_0_3_outputs)
    attn_1_3_output_scores = classifier_weights.loc[
        [("attn_1_3_outputs", str(v)) for v in attn_1_3_outputs]
    ]

    # num_attn_1_0 ####################################################
    def num_predicate_1_0(attn_0_1_output, token):
        if attn_0_1_output in {"3", "1", "0", "5", "4", "2", "<s>"}:
            return token == ""

    num_attn_1_0_pattern = select(tokens, attn_0_1_outputs, num_predicate_1_0)
    num_attn_1_0_outputs = aggregate_sum(num_attn_1_0_pattern, ones)
    num_attn_1_0_output_scores = classifier_weights.loc[
        [("num_attn_1_0_outputs", "_") for v in num_attn_1_0_outputs]
    ].mul(num_attn_1_0_outputs, axis=0)

    # num_attn_1_1 ####################################################
    def num_predicate_1_1(attn_0_0_output, attn_0_2_output):
        if attn_0_0_output in {0, 1, 2, 3, 4, 43}:
            return attn_0_2_output == 0
        elif attn_0_0_output in {34, 11, 44, 5}:
            return attn_0_2_output == 43
        elif attn_0_0_output in {48, 39, 6, 14}:
            return attn_0_2_output == 45
        elif attn_0_0_output in {8, 41, 7}:
            return attn_0_2_output == 40
        elif attn_0_0_output in {40, 9, 29, 15}:
            return attn_0_2_output == 44
        elif attn_0_0_output in {
            32,
            33,
            36,
            10,
            42,
            12,
            13,
            46,
            47,
            16,
            21,
            23,
            25,
            27,
            28,
        }:
            return attn_0_2_output == 49
        elif attn_0_0_output in {17, 18, 19, 22, 24}:
            return attn_0_2_output == 42
        elif attn_0_0_output in {49, 20}:
            return attn_0_2_output == 46
        elif attn_0_0_output in {26, 35, 45, 38}:
            return attn_0_2_output == 47
        elif attn_0_0_output in {37, 30, 31}:
            return attn_0_2_output == 41

    num_attn_1_1_pattern = select(attn_0_2_outputs, attn_0_0_outputs, num_predicate_1_1)
    num_attn_1_1_outputs = aggregate_sum(num_attn_1_1_pattern, num_attn_0_2_outputs)
    num_attn_1_1_output_scores = classifier_weights.loc[
        [("num_attn_1_1_outputs", "_") for v in num_attn_1_1_outputs]
    ].mul(num_attn_1_1_outputs, axis=0)

    # num_attn_1_2 ####################################################
    def num_predicate_1_2(attn_0_1_output, token):
        if attn_0_1_output in {"0"}:
            return token == "0"
        elif attn_0_1_output in {"1"}:
            return token == "1"
        elif attn_0_1_output in {"2"}:
            return token == "2"
        elif attn_0_1_output in {"3"}:
            return token == "3"
        elif attn_0_1_output in {"4"}:
            return token == "4"
        elif attn_0_1_output in {"<s>", "5"}:
            return token == "5"

    num_attn_1_2_pattern = select(tokens, attn_0_1_outputs, num_predicate_1_2)
    num_attn_1_2_outputs = aggregate_sum(num_attn_1_2_pattern, ones)
    num_attn_1_2_output_scores = classifier_weights.loc[
        [("num_attn_1_2_outputs", "_") for v in num_attn_1_2_outputs]
    ].mul(num_attn_1_2_outputs, axis=0)

    # num_attn_1_3 ####################################################
    def num_predicate_1_3(q_token, k_token):
        if q_token in {"3", "1", "0", "5", "4", "2", "<s>"}:
            return k_token == ""

    num_attn_1_3_pattern = select(tokens, tokens, num_predicate_1_3)
    num_attn_1_3_outputs = aggregate_sum(num_attn_1_3_pattern, num_attn_0_3_outputs)
    num_attn_1_3_output_scores = classifier_weights.loc[
        [("num_attn_1_3_outputs", "_") for v in num_attn_1_3_outputs]
    ].mul(num_attn_1_3_outputs, axis=0)

    # mlp_1_0 #####################################################
    def mlp_1_0(attn_1_3_output, attn_1_2_output):
        key = (attn_1_3_output, attn_1_2_output)
        if key in {
            (0, 34),
            (1, 34),
            (2, 34),
            (3, 34),
            (4, 34),
            (5, 34),
            (6, 34),
            (7, 34),
            (8, 34),
            (9, 34),
            (10, 34),
            (11, 34),
            (12, 34),
            (13, 34),
            (14, 34),
            (15, 34),
            (16, 34),
            (17, 34),
            (19, 34),
            (20, 34),
            (22, 34),
            (23, 34),
            (24, 34),
            (25, 34),
            (26, 34),
            (27, 34),
            (28, 34),
            (29, 34),
            (30, 34),
            (31, 34),
            (32, 34),
            (33, 34),
            (34, 34),
            (35, 34),
            (36, 34),
            (37, 34),
            (38, 34),
            (40, 34),
            (41, 34),
            (42, 34),
            (43, 34),
            (44, 34),
            (45, 34),
            (46, 34),
            (47, 34),
            (48, 34),
            (49, 34),
        }:
            return 1
        elif key in {
            (10, 42),
            (10, 46),
            (26, 42),
            (39, 0),
            (39, 3),
            (39, 10),
            (39, 13),
            (39, 19),
            (39, 21),
            (39, 22),
            (39, 25),
            (39, 26),
            (39, 27),
            (39, 33),
            (39, 34),
            (39, 35),
            (39, 39),
            (39, 40),
            (39, 41),
            (39, 42),
            (39, 44),
            (39, 45),
            (39, 46),
            (41, 25),
            (41, 42),
            (41, 45),
            (41, 46),
            (42, 42),
            (42, 46),
            (49, 42),
            (49, 46),
        }:
            return 37
        elif key in {(40, 14), (40, 25), (40, 28), (40, 31), (40, 33), (40, 40)}:
            return 41
        elif key in {(40, 18), (40, 42)}:
            return 42
        return 13

    mlp_1_0_outputs = [
        mlp_1_0(k0, k1) for k0, k1 in zip(attn_1_3_outputs, attn_1_2_outputs)
    ]
    mlp_1_0_output_scores = classifier_weights.loc[
        [("mlp_1_0_outputs", str(v)) for v in mlp_1_0_outputs]
    ]

    # mlp_1_1 #####################################################
    def mlp_1_1(attn_1_3_output, attn_0_2_output):
        key = (attn_1_3_output, attn_0_2_output)
        if key in {
            (0, 7),
            (0, 8),
            (0, 11),
            (0, 12),
            (0, 14),
            (0, 15),
            (0, 18),
            (0, 21),
            (0, 23),
            (0, 25),
            (0, 31),
            (0, 32),
            (0, 38),
            (0, 46),
            (1, 25),
            (2, 8),
            (2, 14),
            (2, 15),
            (2, 18),
            (2, 25),
            (2, 31),
            (2, 32),
            (2, 38),
            (2, 46),
            (5, 25),
            (6, 25),
            (7, 25),
            (8, 15),
            (8, 25),
            (9, 25),
            (10, 25),
            (11, 25),
            (12, 25),
            (13, 25),
            (14, 25),
            (15, 15),
            (15, 25),
            (16, 0),
            (16, 5),
            (16, 6),
            (16, 7),
            (16, 8),
            (16, 9),
            (16, 11),
            (16, 12),
            (16, 14),
            (16, 15),
            (16, 16),
            (16, 17),
            (16, 18),
            (16, 21),
            (16, 22),
            (16, 23),
            (16, 25),
            (16, 27),
            (16, 28),
            (16, 29),
            (16, 31),
            (16, 32),
            (16, 38),
            (16, 40),
            (16, 41),
            (16, 42),
            (16, 43),
            (16, 44),
            (16, 45),
            (16, 46),
            (16, 47),
            (16, 48),
            (16, 49),
            (17, 25),
            (19, 0),
            (19, 5),
            (19, 6),
            (19, 7),
            (19, 8),
            (19, 9),
            (19, 11),
            (19, 12),
            (19, 14),
            (19, 15),
            (19, 16),
            (19, 17),
            (19, 18),
            (19, 21),
            (19, 22),
            (19, 23),
            (19, 25),
            (19, 27),
            (19, 28),
            (19, 29),
            (19, 31),
            (19, 32),
            (19, 38),
            (19, 40),
            (19, 41),
            (19, 42),
            (19, 44),
            (19, 45),
            (19, 46),
            (19, 47),
            (19, 48),
            (19, 49),
            (20, 25),
            (21, 25),
            (22, 25),
            (23, 8),
            (23, 14),
            (23, 15),
            (23, 18),
            (23, 25),
            (23, 31),
            (23, 32),
            (23, 38),
            (24, 25),
            (25, 25),
            (26, 25),
            (27, 7),
            (27, 11),
            (27, 12),
            (27, 14),
            (27, 15),
            (27, 18),
            (27, 21),
            (27, 25),
            (27, 31),
            (27, 32),
            (27, 44),
            (27, 46),
            (28, 15),
            (28, 25),
            (29, 25),
            (31, 25),
            (32, 25),
            (33, 25),
            (34, 25),
            (35, 11),
            (35, 14),
            (35, 15),
            (35, 25),
            (35, 31),
            (35, 38),
            (36, 25),
            (37, 25),
            (38, 8),
            (38, 14),
            (38, 15),
            (38, 18),
            (38, 25),
            (38, 31),
            (38, 32),
            (38, 38),
            (39, 25),
            (40, 7),
            (40, 8),
            (40, 11),
            (40, 12),
            (40, 14),
            (40, 15),
            (40, 18),
            (40, 21),
            (40, 23),
            (40, 25),
            (40, 31),
            (40, 32),
            (40, 38),
            (40, 46),
            (41, 15),
            (41, 25),
            (42, 25),
            (43, 7),
            (43, 8),
            (43, 11),
            (43, 12),
            (43, 14),
            (43, 15),
            (43, 18),
            (43, 21),
            (43, 23),
            (43, 25),
            (43, 31),
            (43, 32),
            (43, 38),
            (43, 46),
            (44, 8),
            (44, 15),
            (44, 25),
            (45, 25),
            (46, 15),
            (46, 25),
            (47, 15),
            (47, 25),
            (48, 8),
            (48, 15),
            (48, 25),
            (49, 8),
            (49, 15),
            (49, 25),
        }:
            return 17
        elif key in {
            (2, 35),
            (14, 35),
            (16, 35),
            (24, 35),
            (26, 0),
            (26, 2),
            (26, 6),
            (26, 7),
            (26, 8),
            (26, 10),
            (26, 12),
            (26, 18),
            (26, 21),
            (26, 26),
            (26, 28),
            (26, 30),
            (26, 32),
            (26, 33),
            (26, 34),
            (26, 35),
            (26, 40),
            (26, 43),
            (26, 44),
            (26, 46),
            (26, 47),
            (29, 35),
            (30, 35),
            (35, 0),
            (35, 2),
            (35, 6),
            (35, 7),
            (35, 8),
            (35, 10),
            (35, 12),
            (35, 18),
            (35, 21),
            (35, 23),
            (35, 24),
            (35, 26),
            (35, 28),
            (35, 30),
            (35, 32),
            (35, 33),
            (35, 34),
            (35, 35),
            (35, 39),
            (35, 40),
            (35, 43),
            (35, 44),
            (35, 45),
            (35, 46),
            (35, 47),
            (36, 18),
            (36, 21),
            (36, 32),
            (36, 35),
            (37, 2),
            (37, 10),
            (37, 18),
            (37, 21),
            (37, 32),
            (37, 35),
            (39, 2),
            (39, 10),
            (39, 18),
            (39, 21),
            (39, 32),
            (39, 35),
            (44, 35),
        }:
            return 45
        elif key in {(27, 8), (27, 38)}:
            return 13
        return 14

    mlp_1_1_outputs = [
        mlp_1_1(k0, k1) for k0, k1 in zip(attn_1_3_outputs, attn_0_2_outputs)
    ]
    mlp_1_1_output_scores = classifier_weights.loc[
        [("mlp_1_1_outputs", str(v)) for v in mlp_1_1_outputs]
    ]

    # num_mlp_1_0 #################################################
    def num_mlp_1_0(num_attn_1_3_output, num_attn_1_0_output):
        key = (num_attn_1_3_output, num_attn_1_0_output)
        return 47

    num_mlp_1_0_outputs = [
        num_mlp_1_0(k0, k1)
        for k0, k1 in zip(num_attn_1_3_outputs, num_attn_1_0_outputs)
    ]
    num_mlp_1_0_output_scores = classifier_weights.loc[
        [("num_mlp_1_0_outputs", str(v)) for v in num_mlp_1_0_outputs]
    ]

    # num_mlp_1_1 #################################################
    def num_mlp_1_1(num_attn_0_1_output):
        key = num_attn_0_1_output
        return 48

    num_mlp_1_1_outputs = [num_mlp_1_1(k0) for k0 in num_attn_0_1_outputs]
    num_mlp_1_1_output_scores = classifier_weights.loc[
        [("num_mlp_1_1_outputs", str(v)) for v in num_mlp_1_1_outputs]
    ]

    # attn_2_0 ####################################################
    def predicate_2_0(q_position, k_position):
        if q_position in {0, 33, 23}:
            return k_position == 22
        elif q_position in {1, 2, 3, 4}:
            return k_position == 4
        elif q_position in {5, 6, 40, 42, 43, 46, 47, 48, 49, 27}:
            return k_position == 5
        elif q_position in {8, 34, 15, 7}:
            return k_position == 6
        elif q_position in {9, 14, 16, 20, 22, 24}:
            return k_position == 7
        elif q_position in {10, 11, 13, 39}:
            return k_position == 8
        elif q_position in {18, 12, 28}:
            return k_position == 10
        elif q_position in {17, 37, 31}:
            return k_position == 11
        elif q_position in {32, 19}:
            return k_position == 9
        elif q_position in {21, 38}:
            return k_position == 20
        elif q_position in {25}:
            return k_position == 13
        elif q_position in {26, 35}:
            return k_position == 25
        elif q_position in {29}:
            return k_position == 24
        elif q_position in {30}:
            return k_position == 21
        elif q_position in {36}:
            return k_position == 16
        elif q_position in {41}:
            return k_position == 43
        elif q_position in {44}:
            return k_position == 14
        elif q_position in {45}:
            return k_position == 17

    attn_2_0_pattern = select_closest(positions, positions, predicate_2_0)
    attn_2_0_outputs = aggregate(attn_2_0_pattern, attn_1_0_outputs)
    attn_2_0_output_scores = classifier_weights.loc[
        [("attn_2_0_outputs", str(v)) for v in attn_2_0_outputs]
    ]

    # attn_2_1 ####################################################
    def predicate_2_1(q_token, k_token):
        if q_token in {"1", "2", "0", "4"}:
            return k_token == "5"
        elif q_token in {"3"}:
            return k_token == "0"
        elif q_token in {"5"}:
            return k_token == "1"
        elif q_token in {"<s>"}:
            return k_token == ""

    attn_2_1_pattern = select_closest(tokens, tokens, predicate_2_1)
    attn_2_1_outputs = aggregate(attn_2_1_pattern, attn_1_1_outputs)
    attn_2_1_output_scores = classifier_weights.loc[
        [("attn_2_1_outputs", str(v)) for v in attn_2_1_outputs]
    ]

    # attn_2_2 ####################################################
    def predicate_2_2(q_attn_0_3_output, k_attn_0_3_output):
        if q_attn_0_3_output in {0, 8}:
            return k_attn_0_3_output == 24
        elif q_attn_0_3_output in {1, 15}:
            return k_attn_0_3_output == 13
        elif q_attn_0_3_output in {27, 2, 3}:
            return k_attn_0_3_output == 6
        elif q_attn_0_3_output in {48, 41, 4}:
            return k_attn_0_3_output == 2
        elif q_attn_0_3_output in {5}:
            return k_attn_0_3_output == 0
        elif q_attn_0_3_output in {17, 6, 22}:
            return k_attn_0_3_output == 14
        elif q_attn_0_3_output in {21, 7}:
            return k_attn_0_3_output == 28
        elif q_attn_0_3_output in {16, 9, 40, 38}:
            return k_attn_0_3_output == 11
        elif q_attn_0_3_output in {10}:
            return k_attn_0_3_output == 30
        elif q_attn_0_3_output in {11}:
            return k_attn_0_3_output == 29
        elif q_attn_0_3_output in {12}:
            return k_attn_0_3_output == 15
        elif q_attn_0_3_output in {13}:
            return k_attn_0_3_output == 27
        elif q_attn_0_3_output in {14}:
            return k_attn_0_3_output == 23
        elif q_attn_0_3_output in {18}:
            return k_attn_0_3_output == 31
        elif q_attn_0_3_output in {19}:
            return k_attn_0_3_output == 5
        elif q_attn_0_3_output in {20, 29, 30}:
            return k_attn_0_3_output == 33
        elif q_attn_0_3_output in {34, 28, 23}:
            return k_attn_0_3_output == 25
        elif q_attn_0_3_output in {24}:
            return k_attn_0_3_output == 20
        elif q_attn_0_3_output in {25}:
            return k_attn_0_3_output == 22
        elif q_attn_0_3_output in {26}:
            return k_attn_0_3_output == 4
        elif q_attn_0_3_output in {33, 49, 31}:
            return k_attn_0_3_output == 17
        elif q_attn_0_3_output in {32}:
            return k_attn_0_3_output == 18
        elif q_attn_0_3_output in {35}:
            return k_attn_0_3_output == 21
        elif q_attn_0_3_output in {36}:
            return k_attn_0_3_output == 8
        elif q_attn_0_3_output in {45, 37, 39}:
            return k_attn_0_3_output == 19
        elif q_attn_0_3_output in {42}:
            return k_attn_0_3_output == 49
        elif q_attn_0_3_output in {43}:
            return k_attn_0_3_output == 16
        elif q_attn_0_3_output in {44}:
            return k_attn_0_3_output == 38
        elif q_attn_0_3_output in {46}:
            return k_attn_0_3_output == 1
        elif q_attn_0_3_output in {47}:
            return k_attn_0_3_output == 12

    attn_2_2_pattern = select_closest(attn_0_3_outputs, attn_0_3_outputs, predicate_2_2)
    attn_2_2_outputs = aggregate(attn_2_2_pattern, attn_0_3_outputs)
    attn_2_2_output_scores = classifier_weights.loc[
        [("attn_2_2_outputs", str(v)) for v in attn_2_2_outputs]
    ]

    # attn_2_3 ####################################################
    def predicate_2_3(q_position, k_position):
        if q_position in {0, 49, 29}:
            return k_position == 12
        elif q_position in {1, 2, 3, 6, 42, 12, 14, 47}:
            return k_position == 7
        elif q_position in {34, 45, 4, 20}:
            return k_position == 11
        elif q_position in {5}:
            return k_position == 33
        elif q_position in {33, 7}:
            return k_position == 9
        elif q_position in {8, 18, 40}:
            return k_position == 3
        elif q_position in {9}:
            return k_position == 17
        elif q_position in {10, 11, 46, 17, 19, 21}:
            return k_position == 24
        elif q_position in {25, 44, 13}:
            return k_position == 36
        elif q_position in {28, 15}:
            return k_position == 1
        elif q_position in {16}:
            return k_position == 29
        elif q_position in {22}:
            return k_position == 26
        elif q_position in {23}:
            return k_position == 37
        elif q_position in {24, 26}:
            return k_position == 39
        elif q_position in {27, 30}:
            return k_position == 14
        elif q_position in {35, 31}:
            return k_position == 18
        elif q_position in {32}:
            return k_position == 34
        elif q_position in {36}:
            return k_position == 21
        elif q_position in {37}:
            return k_position == 30
        elif q_position in {38}:
            return k_position == 5
        elif q_position in {39}:
            return k_position == 2
        elif q_position in {48, 41}:
            return k_position == 15
        elif q_position in {43}:
            return k_position == 22

    attn_2_3_pattern = select_closest(positions, positions, predicate_2_3)
    attn_2_3_outputs = aggregate(attn_2_3_pattern, attn_1_2_outputs)
    attn_2_3_output_scores = classifier_weights.loc[
        [("attn_2_3_outputs", str(v)) for v in attn_2_3_outputs]
    ]

    # num_attn_2_0 ####################################################
    def num_predicate_2_0(num_mlp_1_0_output, token):
        if num_mlp_1_0_output in {
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

    num_attn_2_0_pattern = select(tokens, num_mlp_1_0_outputs, num_predicate_2_0)
    num_attn_2_0_outputs = aggregate_sum(num_attn_2_0_pattern, num_attn_0_3_outputs)
    num_attn_2_0_output_scores = classifier_weights.loc[
        [("num_attn_2_0_outputs", "_") for v in num_attn_2_0_outputs]
    ].mul(num_attn_2_0_outputs, axis=0)

    # num_attn_2_1 ####################################################
    def num_predicate_2_1(attn_0_1_output, token):
        if attn_0_1_output in {"3", "1", "0", "5", "4", "2"}:
            return token == ""
        elif attn_0_1_output in {"<s>"}:
            return token == "3"

    num_attn_2_1_pattern = select(tokens, attn_0_1_outputs, num_predicate_2_1)
    num_attn_2_1_outputs = aggregate_sum(num_attn_2_1_pattern, num_attn_0_1_outputs)
    num_attn_2_1_output_scores = classifier_weights.loc[
        [("num_attn_2_1_outputs", "_") for v in num_attn_2_1_outputs]
    ].mul(num_attn_2_1_outputs, axis=0)

    # num_attn_2_2 ####################################################
    def num_predicate_2_2(attn_1_2_output, attn_0_3_output):
        if attn_1_2_output in {0, 49}:
            return attn_0_3_output == 30
        elif attn_1_2_output in {1, 35, 37, 8, 11}:
            return attn_0_3_output == 45
        elif attn_1_2_output in {2, 41, 45, 15, 27}:
            return attn_0_3_output == 34
        elif attn_1_2_output in {9, 3}:
            return attn_0_3_output == 36
        elif attn_1_2_output in {4}:
            return attn_0_3_output == 23
        elif attn_1_2_output in {16, 40, 5}:
            return attn_0_3_output == 35
        elif attn_1_2_output in {6}:
            return attn_0_3_output == 24
        elif attn_1_2_output in {7}:
            return attn_0_3_output == 32
        elif attn_1_2_output in {10, 36}:
            return attn_0_3_output == 47
        elif attn_1_2_output in {33, 38, 12, 18, 28, 29}:
            return attn_0_3_output == 43
        elif attn_1_2_output in {43, 44, 13, 46, 48, 21, 26}:
            return attn_0_3_output == 37
        elif attn_1_2_output in {30, 14, 22}:
            return attn_0_3_output == 42
        elif attn_1_2_output in {24, 17}:
            return attn_0_3_output == 46
        elif attn_1_2_output in {19, 47, 31}:
            return attn_0_3_output == 48
        elif attn_1_2_output in {34, 20, 42, 39}:
            return attn_0_3_output == 44
        elif attn_1_2_output in {32, 25, 23}:
            return attn_0_3_output == 41

    num_attn_2_2_pattern = select(attn_0_3_outputs, attn_1_2_outputs, num_predicate_2_2)
    num_attn_2_2_outputs = aggregate_sum(num_attn_2_2_pattern, num_attn_0_3_outputs)
    num_attn_2_2_output_scores = classifier_weights.loc[
        [("num_attn_2_2_outputs", "_") for v in num_attn_2_2_outputs]
    ].mul(num_attn_2_2_outputs, axis=0)

    # num_attn_2_3 ####################################################
    def num_predicate_2_3(attn_0_1_output, attn_0_3_output):
        if attn_0_1_output in {"0"}:
            return attn_0_3_output == 39
        elif attn_0_1_output in {"1"}:
            return attn_0_3_output == 36
        elif attn_0_1_output in {"2", "3"}:
            return attn_0_3_output == 35
        elif attn_0_1_output in {"4"}:
            return attn_0_3_output == 44
        elif attn_0_1_output in {"5"}:
            return attn_0_3_output == 24
        elif attn_0_1_output in {"<s>"}:
            return attn_0_3_output == 46

    num_attn_2_3_pattern = select(attn_0_3_outputs, attn_0_1_outputs, num_predicate_2_3)
    num_attn_2_3_outputs = aggregate_sum(num_attn_2_3_pattern, num_attn_1_3_outputs)
    num_attn_2_3_output_scores = classifier_weights.loc[
        [("num_attn_2_3_outputs", "_") for v in num_attn_2_3_outputs]
    ].mul(num_attn_2_3_outputs, axis=0)

    # mlp_2_0 #####################################################
    def mlp_2_0(attn_1_3_output, attn_2_2_output):
        key = (attn_1_3_output, attn_2_2_output)
        if key in {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 15),
            (0, 16),
            (0, 19),
            (0, 20),
            (0, 24),
            (0, 27),
            (0, 29),
            (0, 30),
            (0, 31),
            (0, 34),
            (0, 35),
            (0, 36),
            (0, 38),
            (0, 39),
            (0, 40),
            (0, 41),
            (0, 42),
            (0, 43),
            (0, 44),
            (0, 45),
            (0, 46),
            (0, 47),
            (0, 49),
            (1, 0),
            (1, 1),
            (1, 3),
            (1, 7),
            (1, 16),
            (1, 24),
            (1, 27),
            (1, 29),
            (1, 30),
            (1, 35),
            (1, 36),
            (1, 40),
            (1, 41),
            (1, 42),
            (1, 43),
            (1, 44),
            (1, 45),
            (1, 46),
            (1, 47),
            (1, 49),
            (2, 1),
            (2, 3),
            (2, 24),
            (2, 29),
            (3, 1),
            (3, 3),
            (3, 16),
            (3, 24),
            (3, 29),
            (3, 35),
            (4, 0),
            (4, 1),
            (4, 3),
            (4, 16),
            (4, 24),
            (4, 27),
            (4, 29),
            (4, 35),
            (4, 36),
            (5, 24),
            (6, 0),
            (6, 1),
            (6, 2),
            (6, 3),
            (6, 6),
            (6, 7),
            (6, 8),
            (6, 16),
            (6, 19),
            (6, 24),
            (6, 27),
            (6, 29),
            (6, 30),
            (6, 34),
            (6, 35),
            (6, 36),
            (6, 38),
            (6, 40),
            (6, 41),
            (6, 42),
            (6, 43),
            (6, 44),
            (6, 45),
            (6, 46),
            (6, 47),
            (6, 49),
            (7, 1),
            (7, 3),
            (7, 16),
            (7, 24),
            (7, 29),
            (7, 35),
            (8, 1),
            (8, 3),
            (8, 16),
            (8, 24),
            (8, 29),
            (8, 35),
            (9, 1),
            (9, 3),
            (9, 16),
            (9, 24),
            (9, 29),
            (9, 35),
            (10, 1),
            (10, 3),
            (10, 16),
            (10, 24),
            (10, 29),
            (10, 35),
            (11, 1),
            (11, 3),
            (11, 16),
            (11, 24),
            (11, 29),
            (11, 35),
            (12, 1),
            (12, 3),
            (12, 24),
            (12, 29),
            (13, 1),
            (13, 3),
            (13, 24),
            (13, 29),
            (14, 1),
            (14, 24),
            (15, 1),
            (15, 3),
            (15, 24),
            (15, 29),
            (16, 1),
            (16, 3),
            (16, 16),
            (16, 24),
            (16, 29),
            (16, 35),
            (17, 0),
            (17, 1),
            (17, 2),
            (17, 3),
            (17, 6),
            (17, 7),
            (17, 8),
            (17, 16),
            (17, 19),
            (17, 24),
            (17, 27),
            (17, 29),
            (17, 30),
            (17, 35),
            (17, 36),
            (17, 38),
            (17, 40),
            (17, 41),
            (17, 42),
            (17, 43),
            (17, 44),
            (17, 45),
            (17, 46),
            (17, 47),
            (17, 49),
            (18, 1),
            (18, 3),
            (18, 16),
            (18, 24),
            (18, 29),
            (18, 35),
            (20, 24),
            (21, 1),
            (21, 3),
            (21, 16),
            (21, 24),
            (21, 29),
            (21, 35),
            (22, 0),
            (22, 1),
            (22, 2),
            (22, 3),
            (22, 4),
            (22, 5),
            (22, 6),
            (22, 7),
            (22, 8),
            (22, 9),
            (22, 10),
            (22, 11),
            (22, 12),
            (22, 13),
            (22, 14),
            (22, 15),
            (22, 16),
            (22, 17),
            (22, 18),
            (22, 19),
            (22, 20),
            (22, 21),
            (22, 22),
            (22, 23),
            (22, 24),
            (22, 25),
            (22, 26),
            (22, 27),
            (22, 28),
            (22, 29),
            (22, 30),
            (22, 31),
            (22, 32),
            (22, 33),
            (22, 34),
            (22, 35),
            (22, 36),
            (22, 37),
            (22, 38),
            (22, 39),
            (22, 40),
            (22, 41),
            (22, 42),
            (22, 43),
            (22, 44),
            (22, 45),
            (22, 46),
            (22, 47),
            (22, 48),
            (22, 49),
            (23, 1),
            (23, 3),
            (23, 24),
            (23, 29),
            (24, 1),
            (24, 3),
            (24, 16),
            (24, 24),
            (24, 29),
            (24, 35),
            (25, 0),
            (25, 1),
            (25, 2),
            (25, 3),
            (25, 6),
            (25, 7),
            (25, 8),
            (25, 16),
            (25, 19),
            (25, 24),
            (25, 27),
            (25, 29),
            (25, 30),
            (25, 35),
            (25, 36),
            (25, 38),
            (25, 40),
            (25, 41),
            (25, 42),
            (25, 43),
            (25, 44),
            (25, 45),
            (25, 46),
            (25, 47),
            (25, 49),
            (26, 0),
            (26, 1),
            (26, 3),
            (26, 7),
            (26, 16),
            (26, 24),
            (26, 27),
            (26, 29),
            (26, 30),
            (26, 35),
            (26, 36),
            (26, 41),
            (26, 42),
            (26, 43),
            (26, 44),
            (26, 45),
            (26, 46),
            (26, 47),
            (26, 49),
            (27, 24),
            (28, 0),
            (28, 1),
            (28, 3),
            (28, 16),
            (28, 24),
            (28, 27),
            (28, 29),
            (28, 30),
            (28, 35),
            (28, 36),
            (29, 1),
            (29, 3),
            (29, 16),
            (29, 24),
            (29, 29),
            (29, 35),
            (30, 1),
            (30, 3),
            (30, 16),
            (30, 24),
            (30, 29),
            (30, 35),
            (31, 1),
            (31, 3),
            (31, 24),
            (31, 29),
            (32, 0),
            (32, 1),
            (32, 3),
            (32, 16),
            (32, 24),
            (32, 27),
            (32, 29),
            (32, 35),
            (32, 36),
            (33, 1),
            (33, 24),
            (34, 0),
            (34, 1),
            (34, 3),
            (34, 16),
            (34, 24),
            (34, 27),
            (34, 29),
            (34, 30),
            (34, 35),
            (34, 36),
            (35, 0),
            (35, 1),
            (35, 2),
            (35, 3),
            (35, 6),
            (35, 7),
            (35, 8),
            (35, 16),
            (35, 19),
            (35, 24),
            (35, 27),
            (35, 29),
            (35, 30),
            (35, 35),
            (35, 36),
            (35, 38),
            (35, 40),
            (35, 41),
            (35, 42),
            (35, 43),
            (35, 44),
            (35, 45),
            (35, 46),
            (35, 47),
            (35, 49),
            (36, 1),
            (36, 3),
            (36, 24),
            (36, 29),
            (37, 0),
            (37, 1),
            (37, 3),
            (37, 16),
            (37, 24),
            (37, 27),
            (37, 29),
            (37, 30),
            (37, 35),
            (37, 36),
            (37, 44),
            (37, 47),
            (38, 0),
            (38, 1),
            (38, 3),
            (38, 16),
            (38, 24),
            (38, 29),
            (38, 35),
            (39, 0),
            (39, 1),
            (39, 2),
            (39, 3),
            (39, 4),
            (39, 6),
            (39, 7),
            (39, 8),
            (39, 9),
            (39, 12),
            (39, 13),
            (39, 15),
            (39, 16),
            (39, 17),
            (39, 18),
            (39, 19),
            (39, 20),
            (39, 22),
            (39, 23),
            (39, 24),
            (39, 25),
            (39, 27),
            (39, 29),
            (39, 30),
            (39, 31),
            (39, 34),
            (39, 35),
            (39, 36),
            (39, 38),
            (39, 39),
            (39, 40),
            (39, 41),
            (39, 42),
            (39, 43),
            (39, 44),
            (39, 45),
            (39, 46),
            (39, 47),
            (39, 49),
            (40, 0),
            (40, 1),
            (40, 3),
            (40, 16),
            (40, 24),
            (40, 29),
            (40, 35),
            (41, 0),
            (41, 1),
            (41, 3),
            (41, 16),
            (41, 24),
            (41, 27),
            (41, 29),
            (41, 35),
            (41, 36),
            (42, 0),
            (42, 1),
            (42, 3),
            (42, 16),
            (42, 24),
            (42, 29),
            (42, 35),
            (43, 0),
            (43, 1),
            (43, 3),
            (43, 16),
            (43, 24),
            (43, 29),
            (43, 35),
            (44, 0),
            (44, 1),
            (44, 3),
            (44, 16),
            (44, 24),
            (44, 29),
            (44, 35),
            (45, 0),
            (45, 1),
            (45, 3),
            (45, 16),
            (45, 24),
            (45, 29),
            (45, 35),
            (46, 0),
            (46, 1),
            (46, 3),
            (46, 16),
            (46, 24),
            (46, 29),
            (46, 35),
            (47, 0),
            (47, 1),
            (47, 3),
            (47, 16),
            (47, 24),
            (47, 29),
            (47, 35),
            (48, 0),
            (48, 1),
            (48, 3),
            (48, 16),
            (48, 24),
            (48, 29),
            (48, 35),
            (49, 0),
            (49, 1),
            (49, 3),
            (49, 16),
            (49, 24),
            (49, 29),
            (49, 35),
            (49, 36),
        }:
            return 34
        return 25

    mlp_2_0_outputs = [
        mlp_2_0(k0, k1) for k0, k1 in zip(attn_1_3_outputs, attn_2_2_outputs)
    ]
    mlp_2_0_output_scores = classifier_weights.loc[
        [("mlp_2_0_outputs", str(v)) for v in mlp_2_0_outputs]
    ]

    # mlp_2_1 #####################################################
    def mlp_2_1(attn_1_1_output, attn_2_3_output):
        key = (attn_1_1_output, attn_2_3_output)
        if key in {
            (2, 3),
            (2, 6),
            (2, 29),
            (2, 37),
            (2, 43),
            (2, 45),
            (2, 46),
            (2, 47),
            (2, 49),
            (5, 38),
            (8, 40),
            (10, 7),
            (11, 31),
            (11, 35),
            (14, 3),
            (14, 10),
            (14, 29),
            (14, 34),
            (14, 35),
            (14, 37),
            (14, 43),
            (14, 44),
            (14, 45),
            (14, 46),
            (14, 49),
            (15, 33),
            (17, 7),
            (17, 29),
            (17, 35),
            (17, 37),
            (19, 0),
            (19, 1),
            (19, 2),
            (19, 3),
            (19, 4),
            (19, 5),
            (19, 6),
            (19, 7),
            (19, 9),
            (19, 11),
            (19, 12),
            (19, 13),
            (19, 14),
            (19, 15),
            (19, 16),
            (19, 17),
            (19, 18),
            (19, 19),
            (19, 20),
            (19, 21),
            (19, 22),
            (19, 23),
            (19, 24),
            (19, 25),
            (19, 26),
            (19, 27),
            (19, 29),
            (19, 30),
            (19, 31),
            (19, 32),
            (19, 34),
            (19, 36),
            (19, 37),
            (19, 39),
            (19, 40),
            (19, 41),
            (19, 42),
            (19, 43),
            (19, 44),
            (19, 45),
            (19, 46),
            (19, 47),
            (19, 48),
            (19, 49),
            (20, 31),
            (21, 7),
            (24, 29),
            (24, 34),
            (24, 43),
            (25, 7),
            (26, 31),
            (28, 3),
            (28, 13),
            (28, 28),
            (28, 29),
            (28, 34),
            (28, 37),
            (28, 40),
            (28, 43),
            (28, 44),
            (28, 45),
            (28, 46),
            (28, 49),
            (30, 7),
            (30, 10),
            (30, 29),
            (30, 37),
            (31, 7),
            (32, 35),
            (33, 0),
            (33, 1),
            (33, 2),
            (33, 3),
            (33, 4),
            (33, 5),
            (33, 6),
            (33, 7),
            (33, 9),
            (33, 10),
            (33, 11),
            (33, 12),
            (33, 13),
            (33, 14),
            (33, 15),
            (33, 16),
            (33, 17),
            (33, 18),
            (33, 19),
            (33, 20),
            (33, 21),
            (33, 22),
            (33, 23),
            (33, 24),
            (33, 25),
            (33, 26),
            (33, 27),
            (33, 28),
            (33, 29),
            (33, 30),
            (33, 31),
            (33, 32),
            (33, 33),
            (33, 34),
            (33, 35),
            (33, 36),
            (33, 37),
            (33, 38),
            (33, 39),
            (33, 40),
            (33, 41),
            (33, 42),
            (33, 43),
            (33, 44),
            (33, 45),
            (33, 46),
            (33, 47),
            (33, 48),
            (33, 49),
            (34, 7),
            (34, 38),
            (35, 31),
            (36, 7),
            (37, 7),
            (39, 7),
            (39, 29),
            (39, 37),
            (41, 7),
            (44, 3),
            (44, 10),
            (44, 29),
            (44, 35),
            (44, 37),
            (44, 43),
            (44, 45),
            (45, 1),
            (45, 7),
            (45, 23),
            (47, 7),
            (47, 35),
        }:
            return 38
        elif key in {
            (0, 10),
            (0, 35),
            (0, 38),
            (2, 1),
            (2, 8),
            (2, 10),
            (2, 12),
            (2, 23),
            (2, 28),
            (2, 32),
            (2, 33),
            (2, 35),
            (2, 38),
            (2, 48),
            (4, 10),
            (4, 35),
            (4, 38),
            (5, 10),
            (5, 35),
            (6, 10),
            (6, 35),
            (6, 38),
            (7, 10),
            (7, 35),
            (8, 1),
            (8, 6),
            (8, 7),
            (8, 8),
            (8, 10),
            (8, 12),
            (8, 23),
            (8, 28),
            (8, 32),
            (8, 33),
            (8, 35),
            (8, 38),
            (8, 46),
            (8, 47),
            (8, 48),
            (10, 10),
            (10, 35),
            (10, 38),
            (11, 10),
            (12, 10),
            (12, 35),
            (13, 10),
            (13, 35),
            (15, 8),
            (15, 10),
            (15, 35),
            (15, 38),
            (16, 10),
            (16, 35),
            (16, 38),
            (19, 8),
            (19, 10),
            (19, 28),
            (19, 35),
            (19, 38),
            (21, 10),
            (21, 35),
            (21, 38),
            (22, 10),
            (22, 35),
            (22, 38),
            (23, 10),
            (23, 35),
            (23, 38),
            (24, 1),
            (24, 5),
            (24, 6),
            (24, 7),
            (24, 8),
            (24, 10),
            (24, 12),
            (24, 17),
            (24, 23),
            (24, 27),
            (24, 28),
            (24, 32),
            (24, 33),
            (24, 35),
            (24, 36),
            (24, 38),
            (24, 40),
            (24, 41),
            (24, 42),
            (24, 44),
            (24, 45),
            (24, 46),
            (24, 47),
            (24, 48),
            (24, 49),
            (25, 10),
            (25, 35),
            (25, 38),
            (27, 10),
            (27, 35),
            (31, 10),
            (31, 35),
            (31, 38),
            (32, 10),
            (34, 10),
            (34, 35),
            (35, 10),
            (35, 35),
            (36, 10),
            (36, 35),
            (36, 38),
            (37, 10),
            (37, 35),
            (39, 10),
            (39, 35),
            (39, 38),
            (40, 10),
            (40, 35),
            (40, 38),
            (41, 10),
            (41, 35),
            (41, 38),
            (42, 10),
            (42, 35),
            (43, 10),
            (43, 35),
            (43, 38),
            (45, 8),
            (45, 10),
            (45, 12),
            (45, 28),
            (45, 32),
            (45, 33),
            (45, 35),
            (45, 38),
            (46, 10),
            (46, 35),
            (46, 38),
            (47, 10),
            (48, 10),
            (48, 35),
            (49, 10),
            (49, 35),
            (49, 38),
        }:
            return 19
        elif key in {(2, 7), (17, 10), (19, 33), (28, 10), (28, 35)}:
            return 9
        elif key in {(14, 7), (28, 7), (44, 7)}:
            return 5
        return 2

    mlp_2_1_outputs = [
        mlp_2_1(k0, k1) for k0, k1 in zip(attn_1_1_outputs, attn_2_3_outputs)
    ]
    mlp_2_1_output_scores = classifier_weights.loc[
        [("mlp_2_1_outputs", str(v)) for v in mlp_2_1_outputs]
    ]

    # num_mlp_2_0 #################################################
    def num_mlp_2_0(num_attn_2_3_output, num_attn_1_1_output):
        key = (num_attn_2_3_output, num_attn_1_1_output)
        return 48

    num_mlp_2_0_outputs = [
        num_mlp_2_0(k0, k1)
        for k0, k1 in zip(num_attn_2_3_outputs, num_attn_1_1_outputs)
    ]
    num_mlp_2_0_output_scores = classifier_weights.loc[
        [("num_mlp_2_0_outputs", str(v)) for v in num_mlp_2_0_outputs]
    ]

    # num_mlp_2_1 #################################################
    def num_mlp_2_1(num_attn_2_2_output, num_attn_1_3_output):
        key = (num_attn_2_2_output, num_attn_1_3_output)
        return 45

    num_mlp_2_1_outputs = [
        num_mlp_2_1(k0, k1)
        for k0, k1 in zip(num_attn_2_2_outputs, num_attn_1_3_outputs)
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
            "0",
            "3",
            "2",
            "3",
            "0",
            "2",
            "1",
            "3",
            "5",
            "2",
            "4",
            "4",
            "4",
            "5",
            "3",
        ]
    )
)
