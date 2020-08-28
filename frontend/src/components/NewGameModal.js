import React, {useState} from 'react';
import Modal from 'react-bootstrap/Modal';
import Button from 'react-bootstrap/Button';
import Form from 'react-bootstrap/Form';
import Col from 'react-bootstrap/Col';
import styled from 'styled-components';
import AlertDismissible from './AlertDismissible';

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

const SubmitButtonContainer = styled(Form.Group)`
display: flex;
justify-content: center;
`;

const NewGameModal = (props) => {
  const [playerOne, setPlayerOne] = useState("1");
  const [playerTwo, setPlayerTwo] = useState("3");
  const [showAlert, setShowAlert] = useState(false);
  return (
    <>
    <MyModal
      {...props}
      size="lg"
      aria-labelledby="contained-modal-title-vcenter"
      centered
    >
      <Modal.Header closeButton>
        <Modal.Title id="contained-modal-title-vcenter">
        Select Players
        </Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <Form onSubmit={(e) => {e.preventDefault();
                                if (playerOne === "1" && playerTwo === "1"){
                                  setShowAlert(true);
                                } else {
                                  props.handleClick([playerOne,playerTwo])
                                }}}>
          <Form.Group controlId="formSelectPlayers">
            <Form.Row>
              <Col>
                <Form.Label>Player 1</Form.Label>
                <Form.Control as="select" size="md" defaultValue={playerOne} onChange={(event)=>setPlayerOne(event.target.value)}>
                  <option value={1}>Human</option>
                  <option value={2}>Dense Agent</option>
                  <option value={3}>Alpha0 MCTS Agent</option>
                  <option value={4}>Alpha0 Greedy Agent</option>
                  <option value={5}>Minimax Agent</option>
                  <option value={6}>Negamax Agent</option>
                  <option value={7}>Random Agent</option>
                </Form.Control>
              </Col>
              <Col>
                <Form.Label>Player 2</Form.Label>
                <Form.Control as="select" size="md" defaultValue={playerTwo} onChange={(event)=>setPlayerTwo(event.target.value)}>
                  <option value={1}>Human</option>
                  <option value={2}>Dense Agent</option>
                  <option value={3}>Alpha0 MCTS Agent</option>
                  <option value={4}>Alpha0 Greedy Agent</option>
                  <option value={5}>Minimax Agent</option>
                  <option value={6}>Negamax Agent</option>
                  <option value={7}>Random Agent</option>
                </Form.Control>
              </Col>
            </Form.Row>
          </Form.Group>
          <SubmitButtonContainer>
            <Button variant="info" type="submit">Start Game</Button>
          </SubmitButtonContainer>
        </Form>
        <AlertDismissible show={showAlert} onHide={()=>{setShowAlert(false)}}/>
      </Modal.Body>
      <Modal.Footer>
        <Button variant="danger" onClick={props.onHide}>Cancel</Button>
      </Modal.Footer>
    </MyModal>
    </>
  );
}

export default NewGameModal;