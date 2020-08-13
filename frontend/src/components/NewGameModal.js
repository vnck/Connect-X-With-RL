import React from 'react';
import Modal from 'react-bootstrap/Modal';
import Button from 'react-bootstrap/Button';
import styled from 'styled-components';

const MyModal = styled(Modal)`
  .modal-content {
    background-color: #283747;
    color: white;
  }
  .modal-header {
    border-bottom-color: #1A232E;
  }
  .modal-footer {
    border-top-color: #1A232E;
  }
  .close {
    color: white;
    text-shadow: 0 1px 0 #1A232E;
  }
`;

const ButtonContainer = styled.div`
display: flex;
justify-content: center;
align-items: center;
Button {
  display: block;
  width: 100%;
  margin-left: 1em;
  margin-right: 1em;
}
`;

const NewGameModal = (props) => {
  return (
    <MyModal
      {...props}
      size="lg"
      aria-labelledby="contained-modal-title-vcenter"
      centered
    >
      <Modal.Header closeButton>
        <Modal.Title id="contained-modal-title-vcenter">
        Which player starts first?
        </Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <ButtonContainer>
          <Button variant="info" onClick={() => props.handleClick(1)}>Human</Button>
          <Button variant="info" onClick={() => props.handleClick(2)}>AI</Button>
        </ButtonContainer>
      </Modal.Body>
      <Modal.Footer>
        <Button variant="danger" onClick={props.onHide}>Cancel</Button>
      </Modal.Footer>
    </MyModal>
  );
}

export default NewGameModal;