"""An implementation of the student-proposing instability chaining algorithm."""
from dataclasses import dataclass
from typing import Dict
from typing import Iterable
from typing import List
from typing import Set
from typing import Tuple
from typing import Union
import heapq


def split_by(lst, f):
    trues = []
    falses = []
    for x in lst:
        if f(x):
            trues.append(x)
        else:
            falses.append(x)
    return trues, falses


@dataclass
class Student:
    id: int

    """Preferences on ResidencyProgram.id, from highest priority to lowest priority."""
    preferences: List[int]

    """The highest priority program this student has yet to be rejected from."""
    best_unrejected: int = 0

    def to_apply(self):
        return self.preferences[self.best_unrejected]

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return isinstance(other, Student) and self.id == other.id

    def __str__(self):
        return f"Student({self.id})"

    def __lt__(self, other):
        return self.id < other.id

    def reset_best(self):
        self.best_unrejected = 0


@dataclass
class Couple:
    """A datatype representing a couple, whose preferences are considered jointly."""

    members: Tuple[Student, Student]

    def __post_init__(self):
        self.members = tuple(sorted(self.members))

    def __str__(self):
        return f"({self.members[0]}, {self.members[1]})"

    def __eq__(self, other):
        return isinstance(other, Couple) and self.members == other.members

    def __hash__(self):
        return hash(self.members)

    def reset_best(self):
        for member in self.members:
            member.reset_best()


Applicant = Union[Student, Couple]


@dataclass
class ResidencyProgram:
    id: int

    """Preferences on Student.id, from highest priority to lowest priority."""
    preferences: List[int]

    """The number of open spots."""
    capacity: int

    def select(self, pool: Set[Student]) -> Set[Student]:
        """Select students from `pool` by priority. Return unchosen students."""
        chosen = heapq.nsmallest(
            self.capacity, pool, key=lambda s: self.preferences.index(s.id)
        )
        return pool - set(chosen)

    def __hash__(self):
        return self.id

    def __str__(self):
        return f"Program({self.id})"

    def __eq__(self, other):
        return isinstance(other, ResidencyProgram) and self.id == other.id


@dataclass
class Matching:
    matches: Dict[Student, ResidencyProgram]
    # For simplicity in this Tip, we assume there is no unassigned bucket.

    def reject(self, student):
        if student in self.matches:
            del self.matches[student]
            student.best_unrejected += 1

    def students_matched_to(self, program: ResidencyProgram) -> Set[Student]:
        # Inefficient, but let's keep it simple.
        return set(s for (s, prog) in self.matches.items() if prog.id == program.id)

    def __str__(self):
        return "\n".join(
            [f"{student} -> {program}" for (student, program) in self.matches.items()]
        )


class InstabilityChaining:
    def __init__(self, single_students, couples, programs):
        self.matching = Matching(matches=dict())
        partner_mapping = dict(couple.members for couple in couples)
        self.partner_mapping = partner_mapping | dict(
            (v, k) for k, v in partner_mapping.items()
        )
        self.program_index = {program.id: program for program in programs}

        # Processing couples last reduces the chance of not finding a stable matching.
        self.processing_order: List[Applicant] = single_students + couples

    def run(self) -> Matching:
        for applicant in self.processing_order:
            self.process_one(applicant)
        return self.matching

    def process_one(self, applicant: Applicant) -> None:
        self.applicant_stack = list([applicant])
        self.program_stack: Set[ResidencyProgram] = set()

        while self.applicant_stack or self.program_stack:
            while self.applicant_stack:
                print(f"Applicant stack: {str(self.applicant_stack)}")
                # This could be improved to ensure a couple is always selected
                # over a single student, if a couple is in the stack.
                self.apply(self.applicant_stack.pop())

            print(f"Program stack: {str(self.program_stack)}")
            if self.program_stack:
                program = self.program_stack.pop()
                unstable_applicants = unstable_pairs(
                    program, self.matching, self.partner_mapping, self.program_index
                )
                for applicant in unstable_applicants:
                    applicant.reset_best()
                self.applicant_stack.extend(unstable_applicants)

    def apply(self, applicant: Applicant) -> None:
        if isinstance(applicant, Student):
            (proposer, partner) = (applicant, None)
        elif isinstance(applicant, Couple):
            (proposer, partner) = applicant.members

        def make_pool(student):
            return set([student]) | self.matching.students_matched_to(
                self.program_index[student.to_apply()]
            )

        while proposer.best_unrejected < len(proposer.preferences):
            program = self.program_index[proposer.to_apply()]
            partner_program = (
                self.program_index[partner.to_apply()] if partner else None
            )

            applicants = make_pool(proposer)
            # A special case if both partners prefer the same program. The program
            # needs to prefer both partners and bump two held applications.
            if program == partner_program:
                applicants.add(partner)

            displ = program.select(applicants)
            if partner and program != partner_program:
                displ |= partner_program.select(make_pool(partner))

            if not displ:
                # The program applied to had available positions with no held candidate.
                self.matching.matches[proposer] = program
                if partner:
                    self.matching.matches[partner] = partner_program
                break

            rejected = proposer in displ or partner in displ

            if rejected:
                proposer.best_unrejected += 1
                if partner:
                    partner.best_unrejected += 1
                continue

            self.matching.matches[proposer] = program
            if partner:
                self.matching.matches[partner] = partner_program

            # Not being rejected means that we have to handle all the displaced applicants.
            # This can come in two cases:
            #
            #  - If all displaced applicants are single, continue with one such person as
            #    the new applicant, put the remaining applicants on the applicant stack.
            #
            #  - Each displaced member of a couple has their partner withdraw from their
            #    matched program (if any) and the program is added to the program stack.
            #    Pick one such couple and continue with them applying jointly down their
            #    preference lists, and put the remaining couples on the applicant stack.
            couples, singles = split_by(displ, lambda s: s in self.partner_mapping)
            for bumped in singles:
                self.matching.reject(bumped)

            if len(singles) == len(displ):
                proposer = displ.pop()
                partner = None
                self.applicant_stack.extend(displ)
                continue

            displaced_couples: Set[Couple] = set()
            for bumped in couples:
                withdrawer = self.partner_mapping[bumped]
                couple = Couple(members=(bumped, withdrawer))
                if couple in displaced_couples:
                    continue

                displaced_couples.add(couple)
                self.program_stack.add(self.matching.matches[withdrawer])
                self.matching.reject(bumped)
                self.matching.reject(withdrawer)

            proposer, partner = displaced_couples.pop().members
            self.applicant_stack.extend(displaced_couples)
            self.applicant_stack.extend(singles)


def stable_matching(
    single_students: Iterable[Student],
    couples: Iterable[Couple],
    programs: Iterable[ResidencyProgram],
) -> Matching:
    """Construct a stable matching of students to residency programs.

    Implemented as a variation of the instability chaining algorithm from

        Roth, Alvin E. and Vande Vate, John H.
        "Random Paths to Stability in Two-Sided Matching."
        Econometrica, November 1990, 58(6), pp. 1475–80.

    Note singles and couples require unique ids.

    Arguments:
        single_students: the students to match
        couples: the couples to match
        programs: the ResidencyPrograms to match

    Returns:
        A dict {Student: ResidencyProgram} assigning each student to a program.
    """
    return InstabilityChaining(single_students, couples, programs).run()


def unstable_pairs(
    program: ResidencyProgram,
    matching: Matching,
    partner_mapping: Dict[Student, Student],
    program_index: Dict[int, ResidencyProgram],
) -> Set[Applicant]:
    """Returns all applicants in the market that are unstable with `program`."""

    def make_pool(students, program):
        return set(students) | matching.students_matched_to(program)

    def precedes(L, item1, item2):
        return L.index(item1) < L.index(item2)

    def student_prefers(student, program):
        # Returns true if a student prefers the input program over their
        # assigned program
        try:
            return precedes(
                student.preferences, program.id, matching.matches[student].id
            )
        except ValueError:
            # If the student doesn't have the program in their list, treat as
            # a non-preference.
            return False

    def program_prefers(program, *students):
        # Returns true if a program prefers all input students over some subset
        # of their assigned students.
        displ = program.select(make_pool(students, program))
        return not set(students) & displ

    def joint_mutual_pref(s1, p1, s2, p2):
        s1_existing_or_pref = p1 == matching.matches[s1] or (
            student_prefers(s1, p1) and program_prefers(p1, s1)
        )
        s2_existing_or_pref = p2 == matching.matches[s2] or (
            student_prefers(s2, p2) and program_prefers(p2, s2)
        )

        if s1_existing_or_pref and s2_existing_or_pref:
            # If they're both applying to the same program: they must both be
            # preferred at the same time displaing two candidates.
            return p1 != p2 or program_prefers(p1, s1, s2)

        return False

    all_students = list(matching.matches.keys())
    unstable_applicants: Set[Applicant] = set()

    for student in all_students:
        if (
            program != matching.matches[student]
            and student_prefers(student, program)
            and program_prefers(program, student)
        ):
            applicant: Applicant = student
            if student in partner_mapping:
                # A single student being unstable is not enough if they're part
                # of a couple. They must be joinly unstable with their
                # partner's preferences. I.e., there must be some pair (p1, p2)
                # earlier in their joint list of preferences for which both
                # program p1 prefers student and p2 prefers partner over at
                # least one of their respective matches.
                partner = partner_mapping[student]
                joint_prefs = [
                    (program_index[p1], program_index[p2])
                    for (p1, p2) in zip(student.preferences, partner.preferences)
                ]
                # We only need to consider more preferred pairs in the list
                joint_prefs = joint_prefs[: student.best_unrejected - 1]
                if any(
                    joint_mutual_pref(student, p1, partner, p2)
                    for (p1, p2) in joint_prefs
                ):
                    applicant = Couple(members=(student, partner))
                else:
                    # this couple is not unstable
                    continue

            unstable_applicants.add(applicant)

    return unstable_applicants


def find_unstable_pairs(
    programs: List[ResidencyProgram],
    matching: Matching,
    partner_mapping: Dict[Student, Student],
) -> List[Tuple[Applicant, ResidencyProgram]]:
    program_index = {p.id: p for p in programs}
    return [
        (app, prog)
        for prog in programs
        for app in unstable_pairs(prog, matching, partner_mapping, program_index)
    ]
