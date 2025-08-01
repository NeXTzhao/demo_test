#include "ocp/constraint/constraint_collection.h"

namespace ocp {

ConstraintCollection::ConstraintCollection(const ConstraintCollection &other) =
    default;

ConstraintCollection *ConstraintCollection::clone() const {
  return new ConstraintCollection(*this);
}

size_t ConstraintCollection::GetNumConstraints() const {
  size_t num_of_constraints = 0;
  for (const auto &constraint_term : this->terms_) {
    if (constraint_term->IsActive()) {
      num_of_constraints += constraint_term->GetNumConstraints();
    }
  }
  return num_of_constraints;
}

std::vector<size_t> ConstraintCollection::GetTermsSize() const {
  std::vector<size_t> termsSize(this->terms_.size(), 0);
  for (size_t i = 0; i < this->terms_.size(); ++i) {
    if (this->terms_[i]->IsActive()) {
      termsSize[i] = this->terms_[i]->GetNumConstraints();
    }
  }
  return termsSize;
}

Eigen::VectorXd ConstraintCollection::GetLowerBounds() const {
  const size_t num_of_constraints = GetNumConstraints();
  Eigen::VectorXd lower_bounds =
      Eigen::VectorXd::Constant(num_of_constraints, -1e20);
  size_t offset = 0;
  for (const auto &constraint_term : this->terms_) {
    if (!constraint_term->IsActive()) {
      continue;
    }
    const size_t nc = constraint_term->GetNumConstraints();
    lower_bounds.segment(offset, nc) = constraint_term->GetLowerBounds();
    offset += nc;
  }
  return lower_bounds;
}

Eigen::VectorXd ConstraintCollection::GetUpperBounds() const {
  const size_t num_of_constraints = GetNumConstraints();
  Eigen::VectorXd upper_bounds =
      Eigen::VectorXd::Constant(num_of_constraints, 1e20);
  size_t offset = 0;
  for (const auto &constraint_term : this->terms_) {
    if (!constraint_term->IsActive()) {
      continue;
    }
    const size_t nc = constraint_term->GetNumConstraints();
    upper_bounds.segment(offset, nc) = constraint_term->GetUpperBounds();
    offset += nc;
  }
  return upper_bounds;
}

Eigen::VectorXd
ConstraintCollection::GetValue(const Eigen::Ref<const Eigen::VectorXd> &state,
                               const Eigen::Ref<const Eigen::VectorXd> &input) {
  const size_t num_of_constraints = GetNumConstraints();
  Eigen::VectorXd constraint_values = Eigen::VectorXd::Zero(num_of_constraints);
  size_t offset = 0;
  for (const auto &constraint_term : this->terms_) {
    if (!constraint_term->IsActive()) {
      continue;
    }
    const size_t nc = constraint_term->GetNumConstraints();
    constraint_values.segment(offset, nc) =
        constraint_term->GetValue(state, input);
    offset += nc;
  }
  return constraint_values;
}

void ConstraintCollection::GetLinearApproximation(
    const Eigen::Ref<const Eigen::VectorXd> &state,
    const Eigen::Ref<const Eigen::VectorXd> &input,
    Eigen::Ref<Eigen::MatrixXd> Gx, Eigen::Ref<Eigen::MatrixXd> Gu) const {
  const size_t num_of_constraints = GetNumConstraints();
  Gx = Eigen::MatrixXd::Zero(num_of_constraints, state.size());
  if (input.size() != 0) {
    Gu = Eigen::MatrixXd::Zero(num_of_constraints, input.size());
  }

  size_t offset = 0;
  for (const auto &constraint_term : this->terms_) {
    if (!constraint_term->IsActive()) {
      continue;
    }
    const size_t nc = constraint_term->GetNumConstraints();
    Eigen::MatrixXd tmp_Gx = Eigen::MatrixXd::Zero(nc, state.size());
    Eigen::MatrixXd tmp_Gu;
    if (input.size() != 0) {
      tmp_Gu = Eigen::MatrixXd::Zero(nc, input.size());
    }
    constraint_term->GetLinearApproximation(state, input, tmp_Gx, tmp_Gu);
    Gx.middleRows(offset, nc) = std::move(tmp_Gx);
    if (input.size() != 0) {
      Gu.middleRows(offset, nc) = std::move(tmp_Gu);
    }
    offset += nc;
  }
  return;
}

} // namespace ocp
