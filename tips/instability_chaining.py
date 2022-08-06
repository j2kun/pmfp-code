"""An implementation of the student-proposing instability chaining algorithm."""
from dataclasses import dataclass
from typing import Dict
from typing import Iterable
from typing import List
from typing import Set
from typing import Tuple
import heapq


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
        return self.id


@dataclass
class Couple:
    """A datatype representing a couple, whose preferences are considered jointly."""

    members: Tuple[Student, Student]

    def __str__(self):
        return (self.members[0].id, self.members[1].id)

    def __eq__(self, other):
        return isinstance(other, Couple) and self.members == other.members


Applicant = Student | Couple


@dataclass
class ResidencyProgram:
    id: int
    """Preferences on Student.id, where a lower value implies a higher priority."""
    preferences: Dict[Student, int]
    """The number of open spots."""
    capacity: int

    def select(self, pool: Set[Student]) -> Set[Student]:
        """Select students from `pool` by priority. Return unchosen students."""
        chosen = heapq.nsmallest(self.capacity, pool, key=self.preferences.get)
        return pool - set(chosen)

    def __str__(self):
        return self.id

    def __eq__(self, other):
        return isinstance(other, ResidencyProgram) and self.id == other.id


@dataclass
class Matching:
    matches: Dict[Student, ResidencyProgram]
    # For simplicity in this Tip, we assume there is no unassigned bucket.

    def students_matched_to(self, program: ResidencyProgram) -> Set[Student]:
        # Inefficient, but let's keep it simple.
        return set(s for (s, prog) in self.matches.items() if prog.id == program.id)


def apply(
    applicant: Applicant,
    tentative_matching: Matching,
    partner_mapping: Dict[Student, Student],
    program_index: Dict[int, ResidencyProgram],
) -> Set[Applicant]:
    displaced_applicants: Set[Applicant] = set()
    displaced_programs: Set[ResidencyProgram] = set()

    match applicant:
        case Student() as s:
            (proposer, partner) = (s, None)
        case Couple(members=(proposer, partner)):
            pass

    def make_pool(student):
        return set([student]) | tentative_matching.students_matched_to(
            program_index[student.to_apply()]
        )

    while proposer.best_unrejected < len(proposer.preferences):
        program = program_index[proposer.to_apply()]
        applicants = make_pool(proposer)

        # A special case if both partners prefer the same program. The program
        # needs to prefer both partners and bump two held applications.
        if partner and partner.to_apply() == proposer.to_apply():
            applicants.add(partner)

        displ = program.select(applicants)

        if partner and partner.to_apply() != proposer.to_apply():
            displ |= program_index[partner.to_apply()].select(make_pool(partner))

        rejected = proposer in displ or (partner and partner in displ)

        if rejected:
            proposer.best_unrejected += 1
            if partner:
                partner.best_unrejected += 1
        else:
            for bumped in list(displ.keys()):
                del tentative_matching.matches[bumped]
                # if a member of a couple is bumped, their partner withdraws
                if bumped in partner_mapping and bumped not in (proposer, partner):
                    withdrawer = partner_mapping[bumped]
                    displ.remove(bumped)
                    displ.add(Couple(members=(bumped, withdrawer)))
                    displaced_programs.add(tentative_matching.matches[withdrawer])
                    del tentative_matching.matches[withdrawer]

            tentative_matching[proposer] = program
            if partner:
                tentative_matching[partner] = program_index[partner.to_apply()]

            proposer, partner = next(iter(displ)), None
            proposer.best_unrejected += 1
            if proposer in partner_mapping:
                partner = partner_mapping[proposer]
                partner.best_unrejected += 1
            displaced_applicants.extend(
                displ - set([proposer, Couple(members=(proposer, partner))])
            )

    return displaced_applicants, displaced_programs


def process_one(
    applicant: Applicant,
    tentative_matching: Matching,
    program_index: Dict[int, ResidencyProgram],
    random=None,
) -> None:
    applicant_stack = list([applicant])
    program_stack = list()

    while applicant_stack or program_stack:
        while applicant_stack:
            if random:
                applicant = applicant_stack.pop(random.randrange(len(applicant_stack)))
            else:
                applicant = applicant_stack.pop()

            displaced_students, displaced_programs = apply(
                applicant, tentative_matching, program_index
            )
            applicant_stack.extend(displaced_students)
            program_stack.extend(displaced_programs)

        if program_stack:
            if random:
                program = program_stack.pop(random.randrange(len(program_stack)))
            else:
                program = program_stack.pop()
            applicant_stack.extend(unstable_pairs(program, tentative_matching))


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
    tentative_matching = Matching(matches=dict())
    program_index = {program.id: program for program in programs}

    # Procssing couples last reduces the chance of not finding a stable matching.
    processing_order: List[Applicant] = single_students + couples

    for applicant in processing_order:
        process_one(applicant, tentative_matching, program_index)

    return tentative_matching


def unstable_pairs(
    program: ResidencyProgram,
    matching: Matching,
    partner_mapping: Dict[Student, Student],
) -> Set[Applicant]:
    """Returns all applicants in the market that are unstable with `program`."""

    def make_pool(student):
        return set([student]) | matching.students_matched_to(program)

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

    def program_prefers(program, student):
        # Returns true if a program prefers the input student over at least one
        # of their assigned students.
        return student not in program.select(make_pool(student))

    all_students = list(matching.matches.keys())
    unstable_applicants: Set[Applicant] = set()

    for student in all_students:
        if (
            program != matching.matches[student]
            and student_prefers(student, program)
            and program_prefers(program, student)
        ):
            applicant = student
            if student in partner_mapping:
                applicant = Couple(members=(student, partner_mapping[student]))

            unstable_applicants.add(applicant)

    return unstable_applicants
